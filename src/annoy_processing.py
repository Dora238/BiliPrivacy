from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import TransformersNlpEngine
from presidio_anonymizer import AnonymizerEngine
import argparse
import os

os.environ['HTTP_PROXY'] = 'http://192.168.26.182:4780/'
os.environ['HTTPS_PROXY'] = 'http://192.168.26.182:4780/'
os.environ['FTP_PROXY'] = 'http://192.168.26.182:4780/'

# 从文本文件读取内容
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--user_name', type=str, default='bidao',
                        help="User_name")
    args, _ = parser.parse_known_args()
     
    parser.add_argument('--input_file', type=str, default=f'D:/code/BiliPrivacy/data/processed_data/{args.user_name}.txt',
                        help="Path to the input text file containing user comments.")
    parser.add_argument('--output_file', type=str,
                        default=f'D:/code/BiliPrivacy/data/annoy_processed_data/annoy_{args.user_name}.txt',
                        help="Path to save the privatized comments and BERT scores.")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    with open(args.input_file, 'r', encoding='utf-8') as f:
        text = f.read()


    # Define which transformers model to use
    model_config = [{"lang_code": "zh", "model_name": {
        "spacy": "zh_core_web_sm",  # use a small spaCy model for lemmas, tokens etc.
        "transformers": "ckiplab/bert-base-chinese-ner"
        }
    }]

    # 指定要识别的 PII 实体类型
    # NRP 社保号码
    entities_to_detect = ['DATE_TIME', 'PHONE_NUMBER', 'EMAIL_ADDRESS', 'NRP', 'LOCATION', 'PERSON']


    nlp_engine = TransformersNlpEngine(models=model_config)

    # Set up the engine, loads the NLP module (spaCy model by default)
    # and other PII recognizers
    analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
    # analyzer.load(entities_to_detect)
    # print(analyzer.get_supported_entities())  #

    # Call analyzer to get results
    results = analyzer.analyze(text=text, language='zh', entities=entities_to_detect)
    print(results)

    # Analyzer results are passed to the AnonymizerEngine for anonymization

    anonymizer = AnonymizerEngine()

    anonymized_text = anonymizer.anonymize(text=text, analyzer_results=results)

    print(anonymized_text)

    # 创建目录
    # os.makedirs(f'anonymized_{USER_name}/{USER_name}.txt', exist_ok=True)

    # 第三步：将结果保存到另一个 txt 文件
    with open(args.output_file, 'w', encoding='utf-8') as f:
        f.write(f"匿名化文本:\n\n{anonymized_text}\n\n其他识别结果results{results}")
