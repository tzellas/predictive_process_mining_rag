# predictive_process_mining_ml
Predictive Analysis of Business Processes using Machine Learning Techniques

#docer command 
docker run -d --name qdrant \
  -p 6333:6333 -p 6334:6334 \
  -v "$(pwd)/qdrant_data:/qdrant/storage" \
  qdrant/qdrant
