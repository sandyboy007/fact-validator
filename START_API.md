# Starting the Fact Validator API

## Quick Start

### Terminal 1: Start the API Server
```bash
cd c:\Fact_Validator\services\api
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

**Expected output:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete
```

### Terminal 2: Start the Frontend (in a new terminal)
```bash
cd c:\Fact_Validator\apps\web
npm run dev
```

**Expected output:**
```
> web@0.1.0 dev
> next dev
```

Then open browser to: **http://localhost:3000**

---

## Troubleshooting

### "Address already in use" error
Port 8000 is already taken. Kill it:
```powershell
# Find process on port 8000
Get-NetTCPConnection -LocalPort 8000

# Kill it
Stop-Process -Id <PID> -Force

# Then restart API
```

### "Module not found" error
Make sure virtual environment is activated:
```bash
cd c:\Fact_Validator
.\.venv\Scripts\Activate.ps1
cd services\api
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### "Connection refused" in frontend
- API must be running on port 8000 first
- Frontend looks for API at http://127.0.0.1:8000
- Check terminal for API startup messages

---

## Architecture

- **Frontend:** Next.js at http://localhost:3000
- **API:** FastAPI at http://127.0.0.1:8000
- **Database:** SQLite at c:\Fact_Validator\data\fact_validator.db

Both must be running for the app to work!
