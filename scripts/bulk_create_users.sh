#!/usr/bin/env bash
# One-shot user creation via Supabase Admin REST API.
# Usage:
#   SUPABASE_SERVICE_KEY="eyJ..." bash scripts/bulk_create_users.sh
#
# Get the service key from:
#   Supabase Dashboard → Project Settings → API → service_role (secret)
# DO NOT commit this key. It bypasses Row Level Security.

set -euo pipefail

if [[ -z "${SUPABASE_SERVICE_KEY:-}" ]]; then
  echo "❌ ต้อง set SUPABASE_SERVICE_KEY ก่อนครับ"
  echo "   ดึงจาก: Supabase Dashboard → Settings → API → service_role"
  exit 1
fi

SUPABASE_URL="https://tkauhqdmgggqpxwkhrzm.supabase.co"
PASSWORD="smarthlaw1234"

EMAILS=(
  "mark@smartlaw.th"
  "jason@smartlaw.th"
  "jude@smartlaw.th"
  "wat@smartlaw.th"
  "once@smartlaw.th"
)

for email in "${EMAILS[@]}"; do
  # Capitalize first letter — portable across BSD sed (macOS) and GNU sed
  raw_name="$(echo "$email" | cut -d@ -f1)"
  full_name="$(echo "$raw_name" | awk '{print toupper(substr($0,1,1)) substr($0,2)}')"
  echo "→ สร้าง $email ($full_name) ..."
  response=$(curl -sS -X POST "$SUPABASE_URL/auth/v1/admin/users" \
    -H "Authorization: Bearer $SUPABASE_SERVICE_KEY" \
    -H "apikey: $SUPABASE_SERVICE_KEY" \
    -H "Content-Type: application/json" \
    -d "{\"email\":\"$email\",\"password\":\"$PASSWORD\",\"email_confirm\":true,\"user_metadata\":{\"full_name\":\"$full_name\"}}")
  if echo "$response" | grep -q '"id"'; then
    echo "  ✅ สำเร็จ"
  elif echo "$response" | grep -qi "already.*registered\|already.*exists\|duplicate"; then
    echo "  ⚠️ มีอยู่แล้ว — ข้าม"
  else
    echo "  ❌ ไม่สำเร็จ: $response"
  fi
done

echo ""
echo "🎉 เสร็จสิ้น — login ได้ที่ https://smartlaw.pages.dev ด้วยรหัส: $PASSWORD"
