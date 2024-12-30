import argparse
import jieba
import numpy as np
from annoy import AnnoyIndex
import fasttext
import fasttext.util


def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--user_name', type=str, default='yingshijufeng',
                        help="User_name")
    parser.add_argument('--sensitivity', type=float, default=49.76,
                        help="Sensitivity.")
    parser.add_argument('--delta', type=float, default=1e-05,
                        help="Sensitivity.")
    parser.add_argument('--epsilon', type=float, default=800, help="Privacy parameter for noise addition.")
    args, _ = parser.parse_known_args()
     
    parser.add_argument('--input_file', type=str, default=f'D:/code/BiliPrivacy/data/processed_data/{args.user_name}.txt',
                        help="Path to the input text file containing user comments.")
    parser.add_argument('--output_file', type=str,
                        default=f'D:/code/BiliPrivacy/data/dp_processed_data/dp_{args.user_name}_{args.epsilon}.txt',
                        help="Path to save the privatized comments and BERT scores.")

    parser.add_argument('--embedding-size', type=int, default=300, help="Embedding size (default: 300 for FastText).")
    args = parser.parse_args()
    return args
    


def load_fasttext_model():
    model = fasttext.load_model('D:/code/modelData/fastText/cc.zh.300.bin') # 使用预训练的 FastText 中文模型
    return model


def generate_noise_vector(dimension, epsilon, sensitivity, delta):
    """
    生成基于高斯分布的差分隐私噪声。

    参数：
    - dimension: 噪声向量的维度
    - epsilon: 隐私预算
    - sensitivity: 灵敏度
    - delta: 随机失真参数

    返回：
    - 高斯噪声向量
    """
    #sensitivity = 2 * sigma * num_sigmas * np.sqrt(k)
    mean = 0  # 高斯分布的均值为0
    # 根据差分隐私公式计算标准差
    #Analytic Gaussian mechanism’s noise scale
    scale = np.sqrt((sensitivity**2 / epsilon**2) * 2 * np.log(1.25 / delta))

    # 生成高斯噪声
    gaussian_noise = np.random.normal(mean, scale, size=(dimension,))
    return gaussian_noise


def build_annoy_index(embeddings, embedding_dim):
    ann_index = AnnoyIndex(embedding_dim, 'euclidean')
    for i, vector in enumerate(embeddings):
        ann_index.add_item(i, vector)
    ann_index.build(10)
    return ann_index


def clip_embeddings(embeddings, clip_value=0.94):
    return np.clip(embeddings, -clip_value, clip_value)


def privatize_comment(word_embeddings, comment, model, epsilon, sensitivity, ann_index, delta):
    words = list(jieba.cut(comment.strip()))
    privatized_words = []
    for word in words:
        if word.strip() == "":
            continue
        # 获取词嵌入
        if word in word_embeddings:
            word_embedding = word_embeddings[word]
        else:
            word_embedding = model.get_word_vector(word)

        clipped_embeddings = clip_embeddings(word_embedding, 2)
        noise = generate_noise_vector(len(clipped_embeddings), epsilon, sensitivity, delta)
        noise_embedding = clipped_embeddings + noise

        # 查找最近词语
        nearest_index = ann_index.get_nns_by_vector(noise_embedding, 2, include_distances=False)[1]
        nearest_word = list(word_embeddings.keys())[nearest_index]

        # 如果找不到对应词语，保留原词
        privatized_word = nearest_word
        privatized_words.append(privatized_word)
    return "".join(privatized_words)


def get_word_embeddings(input_file, model):
    word_embeddings = {}
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            comment = line.strip()
            words = list(jieba.cut(comment))
            for word in words:
                if word not in word_embeddings:
                    word_embeddings[word] = model.get_word_vector(word)
    return word_embeddings


if __name__ == '__main__':
    SEED = 42
    np.random.seed(SEED)

    args = parse_args()

    # 加载 FastText 模型
    model = load_fasttext_model()
    embedding_dim = args.embedding_size

    # 获取词嵌入
    word_embeddings = get_word_embeddings(args.input_file, model)
    

    # 创建 Annoy 索引
    f = len(word_embeddings[next(iter(word_embeddings))])
    ann_index = AnnoyIndex(f, 'euclidean')
    for i, (word, embedding) in enumerate(word_embeddings.items()):
        ann_index.add_item(i, embedding)
    ann_index.build(10)

    with open(args.input_file, "r", encoding="utf-8") as f:
        comments = f.readlines()

    # 对评论进行隐私保护
    with open(args.output_file, "w", encoding="utf-8") as output_file:
        for original_comment in comments:
            privatized_comment = privatize_comment(word_embeddings, original_comment, model,
                                                   args.epsilon, args.sensitivity, ann_index, args.delta)

            output_file.write(f"{privatized_comment}\n")

    print("Privatization complete!")
