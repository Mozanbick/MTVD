import os
import argparse
import json
import os
import hashlib

def arg_parse():
    parser = argparse.ArgumentParser(description="Data pre-processing arguments")
    parser.add_argument('--dataset', dest='dataset', help='Dataset to process')
    parser.add_argument('--batch_size', dest='batch_size', help='sample number of each group of selected dataset')
    parser.add_argument('--dataset_dir', dest='dataset_dir', help='Dir of selected dataset')
    parser.add_argument('--output_dir', dest='output_dir', help='Dir of selected dataset')

    parser.set_defaults(
        dataset='BigVul',
        batch_size=5000,
        dataset_dir="../../dataset/Big-Vul",
        output_dir='../input/dataset'
    )

    return parser.parse_args()

def process_dataset(jsonl_path: str, output_base: str) -> None:
    """处理单个数据集文件"""
    dataset_name = os.path.splitext(os.path.basename(jsonl_path))[0]
    output_dir = os.path.join(output_base, dataset_name)
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        file_counter = 0
        current_group = 0
        
        for line in f:
            # 解析JSON数据
            try:
                data = json.loads(line)
                idx = str(data['idx'])
                target = str(data['target'])
                func = data['func']
            except (json.JSONDecodeError, KeyError) as e:
                print(f"格式错误: {e}，跳过该行")
                continue

            # 计算文件哈希
            filehash = hashlib.md5(func.encode()).hexdigest()
            
            # 生成文件名
            filename = f"{idx}_{filehash}_{target}.cpp"
            
            # 确定当前分组
            if file_counter % 5000 == 0:
                current_group = file_counter // 5000
                group_dir = os.path.join(output_dir, f"group{current_group}")
                os.makedirs(group_dir, exist_ok=True)
                print(f"创建分组目录: {group_dir}")
            
            # 写入文件
            output_path = os.path.join(group_dir, filename)
            with open(output_path, 'w', encoding='utf-8') as cpp_file:
                cpp_file.write(func)
            
            file_counter += 1
            
            # 打印进度
            if file_counter % 1000 == 0:
                print(f"已处理 {file_counter} 个文件，当前分组: group{current_group}")

        print(f"数据集 {dataset_name} 处理完成，共生成 {file_counter} 个文件")

def main():
    args = arg_parse()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 处理所有数据集
    for filename in ['train', 'valid', 'test']:
        input_path = os.path.join(args.dataset_dir, f"{filename}.jsonl")
        if not os.path.exists(input_path):
            print(f"警告: 未找到 {input_path}，跳过")
            continue
        
        print(f"\n开始处理数据集: {filename}")
        process_dataset(input_path, args.output_dir)

if __name__ == "__main__":
    main()