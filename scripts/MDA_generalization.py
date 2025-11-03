from _common import *
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
log = logging.getLogger(__name__)

from src.adamerging import ModelWrapper, load_weights, make_functional, softmax_entropy
from src.clip_eval import eval_single_dataset_preprocess_head
from src.heads import get_classification_head
from src.task_vectors import StateDict, TaskVector
from torch.utils.data import DataLoader
from src.datasets.common import maybe_dictionarize
from src.datasets.registry import get_dataset
from clip_checkpoint_path import CHECKPOINT_DIR, finetuned_model_path, pretrained_model_path, sam_retraining_model_path
from src.TSVM_utils import *
from src.task_wise_fusion import check_parameterNamesMatch
from src.utils import num_parameters, timeit_context
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA



# Generate ETF: [C, D]
def generate_etf(d: int, c: int) -> torch.Tensor:
    I = torch.eye(c)
    J = torch.ones((c, c)) / c
    P = I - J
    U, _, _ = torch.linalg.svd(P)
    etf = U[:, :d].T * np.sqrt(c / (c - 1))
    return etf.T  # shape: [C, D]

def optimal_rotation(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    M = B.T @ A
    U, _, Vt = torch.linalg.svd(M)
    R = Vt.T @ U.T
    return R  # shape: [D, D]

def get_class_means(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        feats = []
        labels = []
        for batch in dataloader:
            x = batch['images'].to(device)
            y = batch['labels'].to(device)
            f = model(x)
            feats.append(f)
            labels.append(y)
        feats = F.normalize(torch.cat(feats, dim=0), dim=1)
        labels = torch.cat(labels, dim=0)
        class_means = []
        for c in torch.unique(labels):
            class_feats = feats[labels == c]
            class_means.append(class_feats.mean(dim=0))
        return F.normalize(torch.stack(class_means), dim=1)  # [C, D]

# Neural Collapse Terminus alignment loss
def nct_alignment_loss(features: torch.Tensor, 
                      labels: torch.Tensor, 
                      etf_matrix: torch.Tensor,
                      current_epoch: int,
                      total_epochs: int) -> torch.Tensor:
    features = F.normalize(features, dim=1)
    
    # Ensure labels are within valid range
    assert torch.all(labels < etf_matrix.size(0)), f"Label values exceed ETF matrix dimension {etf_matrix.size(0)}"
    
    targets = etf_matrix[labels]
    
    # Calculate prototype evolution factor (eta in the paper)
    eta = min(1.0, current_epoch / total_epochs)
    
    # If first epoch, initialize with nearest class mean
    if current_epoch == 0:
        # Calculate NCM prototypes
        unique_labels = torch.unique(labels)
        ncm_prototypes = torch.zeros_like(etf_matrix)
        for label in unique_labels:
            class_features = features[labels == label]
            if len(class_features) > 0:
                ncm_prototypes[label] = F.normalize(class_features.mean(dim=0), dim=0)
        
        # Blend between NCM and ETF based on eta
        targets = ncm_prototypes[labels]
    
    return ((features - targets) ** 2).sum(dim=1).mean()


class AdaMerging(torch.nn.Module):
    def __init__(self, paramslist, model, names, exam_datasets, cfg):
        super(AdaMerging, self).__init__()
        self.paramslist = paramslist
        self.model = model
        self.names = names
        self.cfg = cfg
        self.pretrain_lambdas = torch.ones(len(paramslist[0]), 1)
        self.other_lambdas = torch.ones(1, len(paramslist) - 1) * 0.3 
    
        last_layer_lambda = torch.ones(len(paramslist[0]), len(paramslist) - 1) 
        self.last_layer_lambda = torch.nn.Parameter(last_layer_lambda)

        self.classifier = []
        for dataset_name in exam_datasets:
            classification_head = get_classification_head(cfg, dataset_name)
            layer_name = "classifier_{}".format(dataset_name)
            self.add_module(layer_name, classification_head.to(cfg.device))
            self.classifier.append(layer_name)

    def lambdas(self):
        task_lambdas = self.last_layer_lambda
        # task_lambdas = torch.cat([self.other_lambdas, clamped_last_layer], dim=0)
        return torch.cat((self.pretrain_lambdas, task_lambdas), dim=1)

    def collect_trainable_params(self):
        return [self.last_layer_lambda]

    def get_classification_head(self, dataset_name):
        layer_name = "classifier_{}".format(dataset_name)
        classification_head = getattr(self, layer_name)
        return classification_head

    def get_image_encoder(self):
        alph = self.lambdas()
        params = tuple(sum(tuple(pi * lambdasi for pi, lambdasi in zip(p, alph[j].cpu()))) for j, p in enumerate(zip(*self.paramslist)))
        params = tuple(p.cuda(0) for p in params)
        load_weights(self.model, self.names, params)
        return self.model

    def forward(self, inp, dataset_name):
        alph = self.lambdas()
        params = tuple(sum(tuple(pi * lambdasi for pi, lambdasi in zip(p, alph[j].cpu()))) for j, p in enumerate(zip(*self.paramslist)))

        params = tuple(p.cuda(0) for p in params)
        load_weights(self.model, self.names, params)
        feature = self.model(inp)

        layer_name = "classifier_{}".format(dataset_name)
        classification_head = getattr(self, layer_name)
        out = classification_head(feature)
        return out,feature


@hydra.main(config_path=str(CONFIG_DIR), config_name="clip_datasets_generalization_exp1", version_base=None)
    
def main(cfg: DictConfig) -> None:
    cfg.save = str(CHECKPOINT_DIR / cfg.model)
    cfg.data_location = str(DATA_DIR)
    model = cfg.model

    if cfg.sam_retraining:
        log.info("SAM retrained model is used")
        _finetuned_model_path = sam_retraining_model_path
    else:
        _finetuned_model_path = finetuned_model_path

    # exam_datasets = cfg.datasets
    exam_datasets = cfg.seen_datasets
    all_datasets = cfg.seen_datasets + cfg.unseen_datasets
    pretrained_checkpoint = pretrained_model_path(model)

    with timeit_context():
        log.info("load models")
        pretrained_model: nn.Module = torch.load(pretrained_model_path(model))
        task_vectors: List[StateDict] = [
            TaskVector(
                pretrained_checkpoint=pretrained_model_path(model),
                finetuned_checkpoint=_finetuned_model_path(model, dataset_name),
            ).vector
            for dataset_name in exam_datasets
        ]
        check_parameterNamesMatch(task_vectors)

    task_vectors = compute_mda(task_vectors, cfg)

    
    task_vectors = [TaskVector(vector=task_vectors)]

    pretrained_model: nn.Module = torch.load(pretrained_checkpoint)
    pretrained_model_dic: StateDict = pretrained_model.state_dict()

    model = ModelWrapper(pretrained_model, exam_datasets)
    model = model.to(cfg.device)
    _, names = make_functional(model)

    # print("type(task_vectors):", type(task_vectors))
    # task_vector_list = list(task_vectors.values())



    paramslist = []
    paramslist += [tuple(v.detach().requires_grad_(False).cpu() for _, v in pretrained_model_dic.items())]  # pretrain
    paramslist += [tuple(v.detach().requires_grad_(False).cpu() for _, v in tv.vector.items()) for i, tv in enumerate(task_vectors)]  # task vectors
    torch.cuda.empty_cache()
    adamerging_mtl_model = AdaMerging(paramslist, model, names, all_datasets, cfg=cfg)
    with torch.no_grad():
        test_input = torch.randn(1, 3, 224, 224).cuda()  # Assuming standard image size
        feature = adamerging_mtl_model.model(test_input)
        # feature_dim = 512

    rotation_matrices = {
    name: nn.Parameter(torch.eye(feature_dim, device=cfg.device)) 
    for name in exam_datasets
}   


    from src.datasets.common import maybe_dictionarize
    from src.datasets.registry import get_dataset
    
    datasets = {
        dataset_name: get_dataset(
            dataset_name,
            pretrained_model.val_preprocess,
            location=cfg.data_location,
            batch_size=4,
            num_workers=cfg.num_workers,
        )
        for dataset_name in exam_datasets
    }
    
    # Get number of classes for each dataset
    dataset_classes = {name: len(dataset.classnames) for name, dataset in datasets.items()}
    total_classes = sum(dataset_classes.values())
    etf_matrix = generate_etf(feature_dim, total_classes).cuda()
    class_offset_map = {}
    offset = 0
    for name in exam_datasets:
        class_offset_map[name] = offset
        offset += dataset_classes[name]

    print("init lambda:")
    print(adamerging_mtl_model.lambdas())
    print("collect_trainable_params:")
    print(list(adamerging_mtl_model.collect_trainable_params()))

    optimizer = torch.optim.Adam(list(adamerging_mtl_model.collect_trainable_params()), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.0)
    from src.datasets.common import maybe_dictionarize
    from src.datasets.registry import get_dataset

    results = {"epoch": [], "dataset": [], "acc": []}
    if cfg.sam_retraining:
        save_dir = RESULTS_DIR / "sam_retraining" / cfg.model
    else:
        save_dir = RESULTS_DIR / cfg.model
    os.makedirs(save_dir, exist_ok=True)
    results_path: Path = save_dir / "MDA_generalization.csv"

    Total_ACC = 0.0
    for dataset_name in all_datasets:
        image_encoder = adamerging_mtl_model.get_image_encoder()
        classification_head = adamerging_mtl_model.get_classification_head(dataset_name)
        metrics = eval_single_dataset_preprocess_head(image_encoder, classification_head, dataset_name, cfg)
        Total_ACC += metrics["top1"]
        log.info("Eval: Epoch: " + str(0) + " dataset: " + str(dataset_name) + " ACC: " + str(metrics["top1"]))

        results["epoch"].append(0 + 1)
        results["dataset"].append(dataset_name)
        results["acc"].append(metrics["top1"])

    datasets = {
        dataset_name: get_dataset(
            dataset_name, pretrained_model.val_preprocess, location=cfg.data_location, batch_size=16, num_workers=cfg.num_workers
        )
        for dataset_name in exam_datasets
    }
    shuffled_test_loaders: Dict[str, DataLoader] = {dataset_name: dataset.test_loader_shuffle for dataset_name, dataset in datasets.items()}
    shuffled_test_loader_iters = {dataset_name: iter(itertools.cycle(dataloader)) for dataset_name, dataloader in shuffled_test_loaders.items()}
    total_epochs = 1000

    for epoch in tqdm(range(total_epochs)):
        adamerging_mtl_model.train()
        total_loss = 0.0
        all_features = []
        all_labels = []

        alignment_alpha = min(1.0, epoch / 300) 
        rotation_weight = 0.2*alignment_alpha

        for dataset_name in exam_datasets:
            batch = next(shuffled_test_loader_iters[dataset_name])
            batch = maybe_dictionarize(batch)
            x = batch["images"].to(cfg.device)
            y = batch["labels"].to(cfg.device) + class_offset_map[dataset_name]

            with torch.set_grad_enabled(True):
                outputs, feats = adamerging_mtl_model(x, dataset_name) 
                R = rotation_matrices[dataset_name]
                rotated_feats = F.linear(feats, R)
                rotated_feats = F.normalize(rotated_feats, dim=1)

                all_features.append(rotated_feats.detach())
                all_labels.append(y)
                entropy_loss = softmax_entropy(outputs).mean(0)
                align_loss = nct_alignment_loss(rotated_feats, y, etf_matrix, epoch, total_epochs)
                loss = entropy_loss + alignment_alpha * align_loss
                total_loss += loss

        with torch.no_grad():
            features = torch.cat(all_features)
            labels = torch.cat(all_labels)
            class_means = torch.stack([
                F.normalize(features[labels == c].mean(0), dim=0) if (labels == c).any()
                else torch.zeros(feature_dim, device=cfg.device)
                for c in range(total_classes)
            ])
            optimal_R = optimal_rotation(class_means, etf_matrix)

        rotation_reg = sum(((R - optimal_R.detach()) ** 2).mean() for R in rotation_matrices.values())
        total_loss += rotation_weight * rotation_reg

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()


        log.debug(list(adamerging_mtl_model.lambdas().data))

        if ((epoch + 1) % 200) == 0:
            os.makedirs(results_path.parent / os.path.basename(results_path).split(".")[0], exist_ok=True)
            torch.save(
                adamerging_mtl_model.lambdas().data,
                results_path.parent / os.path.basename(results_path).split(".")[0] / os.path.basename(results_path).replace(".csv", ".pt"),
            )

            Total_ACC = 0.0
            for dataset_name in all_datasets:
                image_encoder = adamerging_mtl_model.get_image_encoder()
                classification_head = adamerging_mtl_model.get_classification_head(dataset_name)
                metrics = eval_single_dataset_preprocess_head(image_encoder, classification_head, dataset_name, cfg)
                Total_ACC += metrics["top1"]
                log.info("Eval: Epoch: " + str(epoch) + " dataset: " + str(dataset_name) + " ACC: " + str(metrics["top1"]))

                results["epoch"].append(epoch + 1)
                results["dataset"].append(dataset_name)
                results["acc"].append(metrics["top1"])

            log.info("Eval: Epoch: " + str(epoch) + " Avg ACC:" + str(Total_ACC / len(all_datasets)) + "\n")
            pd.DataFrame(results).to_csv(results_path, index=False)


if __name__ == "__main__":
    main()
