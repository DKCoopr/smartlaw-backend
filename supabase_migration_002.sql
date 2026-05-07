-- ─────────────────────────────────────────────────────────────────
--  SmartLaw — Migration 002: Folders for documents
--  Run AFTER migration_001 in Supabase SQL Editor
-- ─────────────────────────────────────────────────────────────────

-- Add folder column (nullable; NULL = root / no folder)
alter table public.documents
  add column if not exists folder text;

create index if not exists documents_folder_idx
  on public.documents(case_id, folder);

-- Tell PostgREST to reload its schema cache so the new column
-- is visible to the API immediately (prevents PGRST204 errors).
notify pgrst, 'reload schema';
