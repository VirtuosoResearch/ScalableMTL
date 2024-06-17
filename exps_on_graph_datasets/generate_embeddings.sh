# python generate_node2vec_embedding.py --dataset youtube
# python generate_node2vec_embedding.py --dataset dblp
# python generate_node2vec_embedding.py --dataset livejournal
# python generate_deepwalk_embedding.py --dataset youtube
# python generate_deepwalk_embedding.py --dataset dblp
# python generate_deepwalk_embedding.py --dataset livejournal
# python generate_deepwalk_embedding.py --dataset amazon
# python train_bigclam.py --dataset dblp
# python train_bigclam.py --dataset livejournal
# python train_bigclam.py --dataset amazon
# python train_bigclam.py --dataset youtube

python generate_verse_embedding.py --dataset youtube --num_communities 1000
python generate_verse_embedding.py --dataset amazon --num_communities 1000
python generate_verse_embedding.py --dataset dblp --num_communities 1000