-- Convert array embeddings to pgvector type
-- Run this after inserting embeddings from the notebook

-- Cast chunk embeddings from double precision[] to vector
UPDATE ticker_news_chunk_embeddings
SET embedding = embedding::vector
WHERE embedding IS NOT NULL;

-- Cast full document embeddings from double precision[] to vector
UPDATE ticker_news_embeddings
SET embedding = embedding::vector
WHERE embedding IS NOT NULL;

SELECT 'Arrays successfully cast to vector type' AS status;
