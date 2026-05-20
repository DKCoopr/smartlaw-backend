-- ══════════════════════════════════════════════════════════════
-- Thai Immigration Law — pgvector RAG setup
-- Run this in Supabase SQL Editor (Database → SQL Editor)
-- ══════════════════════════════════════════════════════════════

-- 1. Enable pgvector extension
create extension if not exists vector;

-- 2. Law chunks table — stores scraped + chunked law text with embeddings
create table if not exists law_chunks (
  id            uuid primary key default gen_random_uuid(),
  source_url    text not null,
  source_name   text not null,       -- e.g. "ราชกิจจานุเบกษา", "immigration.go.th"
  law_title     text,                -- e.g. "พ.ร.บ. คนเข้าเมือง พ.ศ. 2522"
  section_ref   text,                -- e.g. "มาตรา 12", "ข้อ 5"
  chunk_text    text not null,       -- raw text of this chunk (Thai + EN)
  chunk_index   integer,             -- position within the document
  language      text default 'th',   -- 'th' | 'en'
  category      text,                -- 'visa', 'extension', '90day', 'tm30', 'csoc', 'general'
  embedding     vector(1536),        -- OpenAI text-embedding-3-small dimension
  metadata      jsonb default '{}',  -- extra fields: date, amendment, etc.
  scraped_at    timestamptz default now(),
  created_at    timestamptz default now()
);

-- 3. HNSW index for fast cosine similarity search
create index if not exists law_chunks_embedding_idx
  on law_chunks
  using hnsw (embedding vector_cosine_ops)
  with (m = 16, ef_construction = 64);

-- 4. Indexes for filtering
create index if not exists law_chunks_category_idx on law_chunks(category);
create index if not exists law_chunks_source_idx   on law_chunks(source_name);
create index if not exists law_chunks_lang_idx     on law_chunks(language);

-- 4b. UNIQUE constraint required by embedder.py's upsert(on_conflict="source_url,chunk_index").
-- Without this, re-running the embedder on already-imported chunks raises
-- "no unique or exclusion constraint matching the ON CONFLICT specification".
do $$
begin
  if not exists (
    select 1 from pg_constraint where conname = 'law_chunks_url_idx_unique'
  ) then
    alter table law_chunks add constraint law_chunks_url_idx_unique unique (source_url, chunk_index);
  end if;
end $$;

-- 5. RPC function — vector similarity search with optional category filter
create or replace function search_law_chunks(
  query_embedding vector(1536),
  match_threshold float default 0.70,
  match_count     int   default 8,
  filter_category text  default null
)
returns table (
  id          uuid,
  source_name text,
  law_title   text,
  section_ref text,
  chunk_text  text,
  category    text,
  source_url  text,
  similarity  float
)
language sql stable
as $$
  select
    lc.id,
    lc.source_name,
    lc.law_title,
    lc.section_ref,
    lc.chunk_text,
    lc.category,
    lc.source_url,
    1 - (lc.embedding <=> query_embedding) as similarity
  from law_chunks lc
  where
    (filter_category is null or lc.category = filter_category)
    and 1 - (lc.embedding <=> query_embedding) > match_threshold
  order by lc.embedding <=> query_embedding
  limit match_count;
$$;

-- 6. Scrape log — track which sources have been scraped
create table if not exists law_scrape_log (
  id          uuid primary key default gen_random_uuid(),
  source_name text not null,
  source_url  text,
  status      text default 'pending',  -- 'pending' | 'done' | 'error'
  chunks_added int default 0,
  error_msg   text,
  scraped_at  timestamptz default now()
);
