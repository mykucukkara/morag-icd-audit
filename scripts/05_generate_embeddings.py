import argparse
from loguru import logger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=False)
    parser.add_argument("--top-n", type=int, default=50)
    args = parser.parse_args()
    logger.info("Embeddings generation is integrated into 06_build_faiss_index.py. Please run 06 instead.")

if __name__ == "__main__":
    main()
