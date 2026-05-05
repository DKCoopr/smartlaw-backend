-- ─────────────────────────────────────────────────────────────────
--  SmartLaw — Migration 001
--  Run AFTER supabase_schema.sql in Supabase SQL Editor
--  Adds rich case fields + billing/hearing/transaction tables + storage policies
-- ─────────────────────────────────────────────────────────────────

-- ── Extend cases with rich fields used by frontend ────────────────
alter table public.cases
  add column if not exists case_number       text,
  add column if not exists case_type         text default 'แพ่ง',
  add column if not exists court             text,
  add column if not exists plaintiff_name    text,
  add column if not exists defendant_name    text,
  add column if not exists our_client        text default 'plaintiff',
  add column if not exists claim_amount      numeric default 0,
  add column if not exists ai_strength_score integer,
  add column if not exists assigned_lawyer   text,
  add column if not exists next_hearing      date;

-- Auto-generate case_number on insert
create or replace function set_case_number()
returns trigger as $$
begin
  if new.case_number is null or new.case_number = '' then
    new.case_number := 'SL-' || to_char(now(), 'YYYY-MM') || '-' ||
      upper(substring(replace(new.id::text, '-', ''), 1, 6));
  end if;
  return new;
end;
$$ language plpgsql;

drop trigger if exists cases_set_case_number on public.cases;
create trigger cases_set_case_number
  before insert on public.cases
  for each row execute function set_case_number();

-- ── Hearings ──────────────────────────────────────────────────────
create table if not exists public.hearings (
  id            uuid default gen_random_uuid() primary key,
  case_id       uuid references public.cases(id) on delete cascade not null,
  user_id       uuid references auth.users(id) on delete cascade not null,
  hearing_type  text not null,
  hearing_date  timestamptz not null,
  court_room    text,
  is_completed  boolean default false,
  notes         text,
  created_at    timestamptz default now()
);

create index if not exists hearings_user_id_idx on public.hearings(user_id);
create index if not exists hearings_case_id_idx on public.hearings(case_id);
create index if not exists hearings_date_idx    on public.hearings(hearing_date);

alter table public.hearings enable row level security;
drop policy if exists "hearings: own rows" on public.hearings;
create policy "hearings: own rows" on public.hearings
  for all using (auth.uid() = user_id);

-- ── Billings (invoices/fees) ──────────────────────────────────────
create table if not exists public.billings (
  id              uuid default gen_random_uuid() primary key,
  case_id         uuid references public.cases(id) on delete cascade,
  user_id         uuid references auth.users(id) on delete cascade not null,
  description     text not null,
  amount          numeric not null default 0,
  paid_amount     numeric not null default 0,
  status          text default 'invoiced',  -- invoiced | paid | overdue
  invoice_number  text,
  due_date        date,
  created_at      timestamptz default now(),
  updated_at      timestamptz default now()
);

create index if not exists billings_user_id_idx on public.billings(user_id);
create index if not exists billings_case_id_idx on public.billings(case_id);

alter table public.billings enable row level security;
drop policy if exists "billings: own rows" on public.billings;
create policy "billings: own rows" on public.billings
  for all using (auth.uid() = user_id);

drop trigger if exists billings_updated_at on public.billings;
create trigger billings_updated_at
  before update on public.billings
  for each row execute function update_updated_at();

-- ── Transactions (financial trail in a case) ──────────────────────
create table if not exists public.transactions (
  id            uuid default gen_random_uuid() primary key,
  case_id       uuid references public.cases(id) on delete cascade,
  user_id       uuid references auth.users(id) on delete cascade not null,
  txn_date      date not null,
  from_name     text,
  from_account  text,
  from_bank     text,
  to_name       text,
  to_account    text,
  to_bank       text,
  amount        numeric not null default 0,
  txn_type      text default 'transfer',  -- transfer | suspicious | refund | salary | cash_in | cash_out
  description   text,
  ref_no        text,
  is_flagged    boolean default false,
  created_at    timestamptz default now()
);

create index if not exists transactions_user_id_idx on public.transactions(user_id);
create index if not exists transactions_case_id_idx on public.transactions(case_id);
create index if not exists transactions_date_idx    on public.transactions(txn_date);

alter table public.transactions enable row level security;
drop policy if exists "transactions: own rows" on public.transactions;
create policy "transactions: own rows" on public.transactions
  for all using (auth.uid() = user_id);

-- ── Extend documents with rich fields ─────────────────────────────
alter table public.documents
  add column if not exists doc_label     text,
  add column if not exists original_name text,
  add column if not exists doc_category  text default 'หลักฐาน',
  add column if not exists file_type     text,
  add column if not exists file_size     bigint default 0,
  add column if not exists is_processed  boolean default false,
  add column if not exists ai_summary    text;

-- ── Storage bucket for documents ──────────────────────────────────
insert into storage.buckets (id, name, public)
values ('documents', 'documents', false)
on conflict (id) do nothing;

drop policy if exists "documents bucket: owner read"   on storage.objects;
drop policy if exists "documents bucket: owner write"  on storage.objects;
drop policy if exists "documents bucket: owner update" on storage.objects;
drop policy if exists "documents bucket: owner delete" on storage.objects;

create policy "documents bucket: owner read" on storage.objects
  for select using (bucket_id = 'documents' and (auth.uid())::text = (storage.foldername(name))[1]);

create policy "documents bucket: owner write" on storage.objects
  for insert with check (bucket_id = 'documents' and (auth.uid())::text = (storage.foldername(name))[1]);

create policy "documents bucket: owner update" on storage.objects
  for update using (bucket_id = 'documents' and (auth.uid())::text = (storage.foldername(name))[1]);

create policy "documents bucket: owner delete" on storage.objects
  for delete using (bucket_id = 'documents' and (auth.uid())::text = (storage.foldername(name))[1]);
