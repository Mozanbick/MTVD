import os
import pickle
import shutil
import json
import random
from slice2graph.gen_slice import *
from utils.objects.cpg import Cpg
from utils.objects import FPG
from configs import modelConfig as ModelConfig
from utils.embeddings import Corpus, generate_w2vModel, load_w2vModel
from utils.objects.dataset import GraphDataset

from transformers import AutoTokenizer, AutoModel
import torch


def program_slices_to_graphs(cpg: Cpg, points_file: str, label_path: str):
    fc_points, apu_points, ae_points, fp_points, fr_points = get_points_from_file(points_file=points_file)
    if label_path.endswith('.xml'):
        label_dict = get_vul_lines_from_xml(label_path)
    elif label_path.endswith('.pkl'):
        label_dict = get_vul_lines_from_pkl(label_path)
    elif label_path == "":
        label_dict = {}
    else:
        raise TypeError(f"Label file {label_path} must endswith `.xml` or `.pkl`")
    s1, v1, n1 = FC_slices_to_graphs(cpg, fc_points, label_dict)
    s2, v2, n2 = AUPU_slices_to_graphs(cpg, apu_points, label_dict)
    s3, v3, n3 = AE_slices_to_graphs(cpg, ae_points, label_dict)
    s4, v4, n4 = FP_slices_to_graphs(cpg, fp_points, label_dict)
    s5, v5, n5 = FR_slices_to_graphs(cpg, fr_points, label_dict)
    spg_list = s1 + s2 + s3 + s4 + s5
    save_path = join(ModelConfig.spgs_dir, f"spg_list_{ModelConfig.group}.pkl")
    if not exists(ModelConfig.spgs_dir):
        os.makedirs(ModelConfig.spgs_dir)
    with open(save_path, "wb") as fp:
        pickle.dump(spg_list, fp)
    time.sleep(0.5)
    print(f"vul slices number: {sum([v1, v2, v3, v4, v5])}")
    print(f"non-vul slices number: {sum([n1, n2, n3, n4, n5])}")
    return spg_list


def program_functions_to_graphs(cpg: Cpg):
    fpg_list = []
    vul_count = 0
    non_vul_count = 0
    for testID in cpg.methods:
        for method in cpg.methods[testID]:
            slice_list = list(method.node_id_set)
            items = method.filename.split("@@")
            label = int(items[1])
            assert label == 0 or label == 1
            if label == 1:
                vul_count += 1
            else:
                non_vul_count += 1
            fpg = FPG(testID, method, slice_list, label)
            if len(fpg.node_list) == 0:
                continue
            fpg_list.append(fpg)
    save_path = join(ModelConfig.fpgs_dir, f"fpg_list_{ModelConfig.group}.pkl")
    if not exists(ModelConfig.fpgs_dir):
        os.makedirs(ModelConfig.fpgs_dir)
    with open(save_path, "wb") as fp:
        pickle.dump(fpg_list, fp)
    time.sleep(0.5)
    print(f"vul functions number: {vul_count}")
    print(f"non-vul functions number: {non_vul_count}")
    return fpg_list


def program_slices_to_graphs_with_load():
    save_path = join(ModelConfig.spgs_dir, f"spg_list_{ModelConfig.group}.pkl")
    with open(save_path, "rb") as fp:
        spg_list = pickle.load(fp)
    return spg_list


def program_functions_to_graphs_with_load():
    save_path = join(ModelConfig.fpgs_dir, f"fpg_list_{ModelConfig.group}.pkl")
    with open(save_path, "rb") as fp:
        fpg_list = pickle.load(fp)
    return fpg_list


def program_slices_to_graphs_load_all():
    spg_list = []
    for file in os.listdir(ModelConfig.spgs_dir):
        path = os.path.join(ModelConfig.spgs_dir, file)
        with open(path, "rb") as fp:
            glist = pickle.load(fp)
        spg_list += glist
    return spg_list


def program_functions_to_graphs_load_all():
    fpg_list = []
    for file in os.listdir(ModelConfig.fpgs_dir):
        path = join(ModelConfig.fpgs_dir, file)
        with open(path, "rb") as fp:
            glist = pickle.load(fp)
        fpg_list += glist
    return fpg_list


def program_slices_to_graphs_load_test():
    spg_list = []
    for file in os.listdir(ModelConfig.spgs_dir):
        if 'test' in file:
            path = join(ModelConfig.spgs_dir, file)
            with open(path, "rb") as fp:
                glist = pickle.load(fp)
            spg_list += glist
    return spg_list


def graph_to_dataset(
        cpg: Cpg,
        points_file: str,
        label_path: str,
        corpus_path: str,
        w2v_path: str,
        save_path: str
):
    """
    ++ train embedding nn
    ++ embed nodes
    ++ convert spg into graph dataset
    """
    # spg_list = program_slices_to_graphs(cpg, points_file, label_path)
    spg_list = program_slices_to_graphs_with_load()
    # spg_list = program_functions_to_graphs_with_load()
    # spg_list = program_functions_to_graphs_load_all()
    # spg_list = program_slices_to_graphs_load_test()
    # generate corpus
    # corpus = Corpus(corpus_path)
    # for spg in tqdm(spg_list, desc="generating corpus"):
    #     corpus.add_corpus(spg.node_list)
    # corpus.save()
    # generate w2v_model
    # w2v = generate_w2vModel(corpus_path, w2v_path, size=ModelConfig.embed_dim)
    w2v = load_w2vModel(w2v_path)
    # embed and save
    mode = True if 'test' in ModelConfig.group else False
    dataset = GraphDataset(ModelConfig.dataset, save_path, test=mode)
    for spg in spg_list:
        spg.embed(ModelConfig.nodes_dim, w2v.wv)
        # print(spg)
        dataset.add_graph(spg)
    dataset.save()


def change_test_paths(dataset: str, save_path: str):
    """
    The original path of test dataset is `<save_path>/<dataset_name>/<vul_type>/dataset.bin`,
    we need to convert this path to `<save_path>/<vul_type>/<dataset_name>/dataset.bin`,
    for the convenience for the model test implementation.
    """
    # check sub folder structure
    if dataset not in os.listdir(save_path):
        return
    for vul_type in os.listdir(join(save_path, dataset)):
        for filename in os.listdir(join(save_path, dataset, vul_type)):
            src_path = join(save_path, dataset, vul_type, filename)
            dst_dir = join(save_path, vul_type, dataset)
            if not exists(dst_dir):
                os.makedirs(dst_dir)
            shutil.move(src_path, join(dst_dir, filename))
    # delete original folder
    shutil.rmtree(join(save_path, dataset))

# 假设的向量数据库结构示例（需要实际填充）
VECTOR_DB = [
    {
        "vulnerable": "void insecure_function() {\n    char buffer[10];\n    gets(buffer);\n}",
        "patch": "void secure_function() {\n    char buffer[10];\n    fgets(buffer, 10, stdin);\n}",
        "vuln_vector": [0.1, 0.2, 0.3],  # 实际应使用代码嵌入向量
        "patch_vector": [0.4, 0.5, 0.6]
    }
]

def get_code_vector(code: str) -> List[float]:
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = AutoModel.from_pretrained("microsoft/codebert-base")
    inputs = tokenizer(code, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """计算余弦相似度（示例实现）"""
    dot = sum(a*b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a**2 for a in vec_a)**0.5
    norm_b = sum(b**2 for b in vec_b)**0.5
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0

def find_most_similar_pair(target_vector: List[float]) -> dict:
    """在向量数据库中查找最相似代码对"""
    best_pair = None
    max_similarity = -1
    
    for pair in VECTOR_DB:
        # 计算与漏洞代码的相似度
        sim_vuln = cosine_similarity(target_vector, pair["vuln_vector"])
        # 计算与补丁代码的相似度
        sim_patch = cosine_similarity(target_vector, pair["patch_vector"])
        # 取两者中的最大值作为当前对的相似度
        current_max = max(sim_vuln, sim_patch)
        
        if current_max > max_similarity:
            max_similarity = current_max
            best_pair = pair
            
    return best_pair


def graph_to_dataset_new(
        cpg: Cpg,
        points_file: str,
        label_path: str,
        corpus_path: str,
        w2v_path: str,
        save_path: str,
        args
):
    # generate slice/function program slice
    if not args.func_level:  # slice level
        if args.gen_graph:
            g_list = program_slices_to_graphs(cpg, points_file, label_path)
        else:
            try:
                g_list = program_slices_to_graphs_with_load()
            except FileNotFoundError:
                return
    else:  # function level
        if args.gen_graph:
            g_list = program_functions_to_graphs(cpg)
        else:
            try:
                g_list = program_functions_to_graphs_with_load()
            except FileNotFoundError:
                return
    # generate corpus
    if args.gen_corpus:
        corpus = Corpus(corpus_path)
        if not args.func_level:  # slice level
            all_list = program_slices_to_graphs_with_load()
        else:  # function level
            all_list = program_functions_to_graphs_with_load()
        for g in tqdm(all_list, desc="generating corpus..."):
            method = cpg.get_method_by_filename(g.testID, g.filenames.pop())
            corpus.add_corpus(g.node_list, method)
        corpus.save()
    # generate embedding model
    if args.gen_w2v:
        generate_w2vModel(corpus_path, w2v_path, size=ModelConfig.embed_dim)
    # convert slice/function program graph to model input dataset
    if args.g2dataset:
        # # load embedding model
        # w2v = load_w2vModel(w2v_path)
        # # embed and save
        # mode = True if 'test' in ModelConfig.group else False
        # dataset = GraphDataset(ModelConfig.dataset, save_path, test=mode)
        # for spg in tqdm(g_list, desc="convert graphs to dataset..."):
        #     method = cpg.get_method_by_filename(spg.testID, spg.filenames.pop())
        #     spg.embed(ModelConfig.nodes_dim, method, w2v.wv)
        #     dataset.add_graph(spg)
        # dataset.save()
        # if mode:
        #     change_test_paths(ModelConfig.dataset, save_path)
        mode = True if 'test' in ModelConfig.group else False
        if not exists(save_path):
            os.mkdir(save_path)
        # vulnerability prediction subtask
        vul_pre = []
        for spg in tqdm(g_list, desc="convert graphs to dataset 1 ..."):
            method = cpg.get_method_by_filename(spg.testID, spg.filenames.pop())
            item = {
                "instruction": "Determine if a given code function has a potential vulnerability",
                "input": method._code,
                "output": method._label
            }
            vul_pre.append(item)
        with open(join(save_path, f"ft_vulnerability_pre_{ModelConfig.group}.json"), 'w') as fp:
            json.dump(vul_pre, fp)
        if mode:  # just vulnerability prediction subtask for test dataset
            return
        # dependency prediction subtask
        dep_pre = []
        selected_count = max(1, int(0.1 * len(g_list)))
        selected_cpgs = random.sample(g_list, k=selected_count)
        for spg in tqdm(selected_cpgs, desc="convert graphs to dataset 2 ..."):
            method = cpg.get_method_by_filename(spg.testID, spg.filenames.pop())
            nodes = list(method.nodes.values())
            # 过滤掉没有足够节点的CPG
            if len(nodes) < 2:
                continue
            # 生成10个数据项
            for _ in range(10):
                # 随机选取两个不同节点
                node1, node2 = random.sample(nodes, k=2)
                # 构建输入内容
                code1 = getattr(node1, 'code', '[No code available]')
                code2 = getattr(node2, 'code', '[No code available]')
                input_content = (
                    f"Function code:\n{method._code}\n\n"
                    f"Code element 1:\n{code1}\n\n"
                    f"Code element 2:\n{code2}"
                )
                # 检查依赖关系
                dependencies = []
                # AST依赖检查
                if (node1.id in method.ast_edges and node2.id in method.ast_edges[node1.id]) or \
                (node2.id in method.ast_edges and node1.id in method.ast_edges[node2.id]):
                    dependencies.append("ast")
                # CFG依赖检查
                if (node1.id in method.cfg_edges and node2.id in method.cfg_edges[node1.id]) or \
                (node2.id in method.cfg_edges and node1.id in method.cfg_edges[node2.id]):
                    dependencies.append("cfg")
                # CDG依赖检查
                if (node1.id in method.cdg_edges and node2.id in method.cdg_edges[node1.id]) or \
                (node2.id in method.cdg_edges and node1.id in method.cdg_edges[node2.id]):
                    dependencies.append("cdg")
                # DDG依赖检查
                if (node1.id in method.ddg_edges and node2.id in method.ddg_edges[node1.id]) or \
                (node2.id in method.ddg_edges and node1.id in method.ddg_edges[node2.id]):
                    dependencies.append("ddg")
                
                # 构建输出内容
                if dependencies:
                    output_content = f"yes, {', '.join(dependencies)} edges"
                else:
                    output_content = "no"
                
                # 添加完整数据项
                dep_pre.append({
                    "instruction": "Given a code function and two code elements in it, "
                                "determine if there is a dependency between the code elements, "
                                "and if so give the specific dependency.",
                    "input": input_content,
                    "output": output_content
                })
        with open(join(save_path, f"ft_dependency_pre_{ModelConfig.group}.json"), 'w') as fp:
            json.dump(dep_pre, fp)
        # patch prediction subtask
        pat_pre = []
        for spg in tqdm(g_list, desc="convert graphs to dataset 3 ..."):
            method = cpg.get_method_by_filename(spg.testID, spg.filenames.pop())
            try:
                # 获取目标函数代码
                target_code = method._code
                target_vector = get_code_vector(target_code)
                
                # 查找最相似代码对
                similar_pair = find_most_similar_pair(target_vector)
                
                # 构建输入内容
                input_content = (
                    f"Target function:\n{target_code}\n\n"
                    f"Similar vulnerable function:\n{similar_pair['vulnerable']}\n"
                    f"Similar patch function:\n{similar_pair['patch']}"
                )
                
                # 计算相似度确定输出
                sim_vuln = cosine_similarity(target_vector, similar_pair["vuln_vector"])
                sim_patch = cosine_similarity(target_vector, similar_pair["patch_vector"])
                output = "patched" if sim_patch > sim_vuln else "vulnerable"
                
                # 添加数据项
                pat_pre.append({
                    "instruction": "Given the target function code, and a pair of similarly vulnerable function code and patch function code, determine whether the target function has been patched.",
                    "input": input_content,
                    "output": output
                })
                
            except Exception as e:
                print(f"Error processing CPG: {str(e)}")
                continue
        with open(join(save_path, f"ft_patch_pre_{ModelConfig.group}.json"), 'w') as fp:
            json.dump(pat_pre, fp)
