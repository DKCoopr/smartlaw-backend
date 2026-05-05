-- ─────────────────────────────────────────────────────────────────
--  SmartLaw — Supabase Database Schema
--  Run this in: Supabase Dashboard → SQL Editor → New Query
-- ─────────────────────────────────────────────────────────────────

-- Enable pgvector for future RAG/embedding search
create extension if not exists vector;

-- ── Users profile (extends Supabase auth.users) ──────────────────
create table if not exists public.profiles (
  id          uuid references auth.users(id) on delete cascade primary key,
  full_name   text,
  role        text default 'lawyer',   -- lawyer | officer | admin
  station     text,
  badge_no    text,
  created_at  timestamptz default now(),
  updated_at  timestamptz default now()
);

-- ── Cases ─────────────────────────────────────────────────────────
create table if not exists public.cases (
  id          uuid default gen_random_uuid() primary key,
  user_id     uuid references auth.users(id) on delete cascade not null,
  title       text not null,
  status      text default 'open',    -- open | closed | deleted
  transcript  text,
  form_data   jsonb default '{}',
  analysis    jsonb default '{}',
  created_at  timestamptz default now(),
  updated_at  timestamptz default now()
);

-- ── Documents ─────────────────────────────────────────────────────
create table if not exists public.documents (
  id           uuid default gen_random_uuid() primary key,
  case_id      uuid references public.cases(id) on delete cascade,
  user_id      uuid references auth.users(id) on delete cascade not null,
  title        text not null,
  doc_type     text default 'police_daily_log',
  storage_path text,                  -- Supabase Storage path
  created_at   timestamptz default now()
);

-- ── Audit log (immutable — never delete rows) ─────────────────────
create table if not exists public.audit_log (
  id          bigserial primary key,
  user_id     uuid references auth.users(id),
  action      text not null,          -- e.g. "case.create", "transcribe"
  resource_id text,
  metadata    jsonb default '{}',
  ip_address  inet,
  created_at  timestamptz default now()
);

-- ── Row Level Security ────────────────────────────────────────────
alter table public.profiles  enable row level security;
alter table public.cases      enable row level security;
alter table public.documents  enable row level security;
alter table public.audit_log  enable row level security;

-- Users can only see their own profile
create policy "profiles: own row" on public.profiles
  for all using (auth.uid() = id);

-- Users can only see their own cases
create policy "cases: own rows" on public.cases
  for all using (auth.uid() = user_id);

-- Users can only see their own documents
create policy "documents: own rows" on public.documents
  for all using (auth.uid() = user_id);

-- Audit log: insert only, no reads for regular users
create policy "audit_log: insert only" on public.audit_log
  for insert with check (auth.uid() = user_id);

-- ── Indexes ───────────────────────────────────────────────────────
create index if not exists cases_user_id_idx    on public.cases(user_id);
create index if not exists cases_status_idx     on public.cases(status);
create index if not exists cases_created_at_idx on public.cases(created_at desc);
create index if not exists documents_case_id_idx on public.documents(case_id);

-- ── Auto-update updated_at ────────────────────────────────────────
create or replace function update_updated_at()
returns trigger as $$
begin
  new.updated_at = now();
  return new;
end;
$$ language plpgsql;

create trigger cases_updated_at
  before update on public.cases
  for each row execute function update_updated_at();

create trigger profiles_updated_at
  before update on public.profiles
  for each row execute function update_updated_at();
