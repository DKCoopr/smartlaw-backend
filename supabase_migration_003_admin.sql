-- ─────────────────────────────────────────────────────────────────
--  SmartLaw — Migration 003: Admin Panel
--  Run this in: Supabase Dashboard → SQL Editor → New Query
--
--  What this does:
--   1. Creates error_logs table for frontend bug capture
--   2. Adds admin RLS policies so admins can read across users
--   3. Auto-creates a profiles row when a new auth.users is inserted
--   4. Promotes dean@smartlaw.th to admin (idempotent)
-- ─────────────────────────────────────────────────────────────────

-- ── 1. error_logs ────────────────────────────────────────────────
create table if not exists public.error_logs (
  id          bigserial primary key,
  user_id     uuid references auth.users(id) on delete set null,
  level       text default 'error',         -- error | warn | info
  message     text not null,
  stack       text,
  url         text,
  user_agent  text,
  context     jsonb default '{}',
  created_at  timestamptz default now()
);

create index if not exists error_logs_created_at_idx on public.error_logs(created_at desc);
create index if not exists error_logs_level_idx      on public.error_logs(level);
create index if not exists error_logs_user_id_idx    on public.error_logs(user_id);

alter table public.error_logs enable row level security;

-- Anyone authenticated can insert their own error
drop policy if exists "error_logs: insert own" on public.error_logs;
create policy "error_logs: insert own" on public.error_logs
  for insert with check (auth.uid() = user_id);

-- Only admins can read/delete
drop policy if exists "error_logs: admin read" on public.error_logs;
create policy "error_logs: admin read" on public.error_logs
  for select using (
    exists (select 1 from public.profiles p where p.id = auth.uid() and p.role = 'admin')
  );

drop policy if exists "error_logs: admin delete" on public.error_logs;
create policy "error_logs: admin delete" on public.error_logs
  for delete using (
    exists (select 1 from public.profiles p where p.id = auth.uid() and p.role = 'admin')
  );

-- ── 2. Admin policies on existing tables ─────────────────────────
-- Admins can read all profiles (for the user list)
drop policy if exists "profiles: admin read all" on public.profiles;
create policy "profiles: admin read all" on public.profiles
  for select using (
    exists (select 1 from public.profiles p where p.id = auth.uid() and p.role = 'admin')
  );

-- Admins can update any profile (role / full_name)
drop policy if exists "profiles: admin update all" on public.profiles;
create policy "profiles: admin update all" on public.profiles
  for update using (
    exists (select 1 from public.profiles p where p.id = auth.uid() and p.role = 'admin')
  );

-- Admins can insert profile rows for new users
drop policy if exists "profiles: admin insert all" on public.profiles;
create policy "profiles: admin insert all" on public.profiles
  for insert with check (
    exists (select 1 from public.profiles p where p.id = auth.uid() and p.role = 'admin')
  );

-- ── 3. Auto-create profile on signup ─────────────────────────────
-- Without this, new users have no profile row → role lookup fails
create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer set search_path = public
as $$
begin
  insert into public.profiles (id, full_name, role)
  values (
    new.id,
    coalesce(new.raw_user_meta_data->>'full_name', split_part(new.email, '@', 1)),
    'lawyer'
  )
  on conflict (id) do nothing;
  return new;
end;
$$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
  after insert on auth.users
  for each row execute function public.handle_new_user();

-- ── 4. Backfill missing profiles for existing users ──────────────
insert into public.profiles (id, full_name, role)
select
  u.id,
  coalesce(u.raw_user_meta_data->>'full_name', split_part(u.email, '@', 1)),
  'lawyer'
from auth.users u
left join public.profiles p on p.id = u.id
where p.id is null;

-- ── 5. Promote dean@smartlaw.th to admin ─────────────────────────
update public.profiles
set role = 'admin'
where id = (select id from auth.users where email = 'dean@smartlaw.th');

-- Verify
select u.email, p.role, p.full_name
from auth.users u
join public.profiles p on p.id = u.id
order by (p.role = 'admin') desc, u.email;
