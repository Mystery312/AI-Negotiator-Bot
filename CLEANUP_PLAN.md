# Codebase Cleanup Plan

## Current Issues

1. **Duplicate Documentation**: Multiple README-style files in root
2. **Scattered Scripts**: Setup/utility scripts not organized
3. **Redundant Files**: Multiple overlapping documentation files
4. **Cache/Temp Files**: Untracked cache and temporary data
5. **Mixed Structure**: Backend and frontend code in separate directories

## Files to Consolidate/Remove

### Documentation Files (Root Level)
- ❌ CLEANUP_REPORT.md (old, superseded by newer docs)
- ❌ DEMO_READY.md (redundant with RUN_DEMO.md)
- ❌ INTEGRATION_COMPLETE.md (archived information)
- ❌ INTEGRATION_QUICK_START.md (merged into README)
- ❌ FRONTEND_BACKEND_INTEGRATION_GUIDE.md (62KB, move to docs/)
- ✅ README.md (KEEP - main entry point)
- ✅ QUICK_START.md (KEEP - essential quick reference)
- ✅ RUN_DEMO.md (KEEP - launch instructions)
- ✅ START_HERE_NEW.md → Rename to START_HERE.md
- ✅ DOND_DATASET_SETUP.md (KEEP - essential for dataset)
- ✅ RESTORATION_COMPLETE.md → Move to docs/archive/

### Scripts (Root Level)
- ✅ create_sample_dond_data.py → Move to scripts/
- ✅ setup_dond_dataset.py → Move to scripts/
- ✅ test_app.py → Move to scripts/
- ✅ start_demo.sh (KEEP - essential)
- ✅ stop_demo.sh (KEEP - essential)

### Directories to Clean
- ✅ cache/ → Add to .gitignore, clean contents
- ✅ data/ → Keep structure, remove old temp files
- ✅ docs/ → Archive old files, keep essential

## Final Structure

```
chatbot/
├── README.md                          # Main entry point
├── QUICK_START.md                     # Quick start guide
├── RUN_DEMO.md                        # How to run the demo
├── START_HERE.md                      # Getting started (renamed)
├── DOND_DATASET_SETUP.md             # Dataset setup guide
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
├── start_demo.sh                      # Demo launcher
├── stop_demo.sh                       # Demo stopper
│
├── scripts/                           # Utility scripts
│   ├── create_sample_dond_data.py
│   ├── setup_dond_dataset.py
│   └── test_app.py
│
├── docs/                              # Documentation
│   ├── NEGOTIATION_CHATBOT.md        # Chatbot guide
│   ├── HOW_TO_RUN.md                 # Detailed run guide
│   ├── RESOURCE_MANAGEMENT_SYSTEM_PLAN.md
│   └── archive/                       # Archived docs
│       ├── CLEANUP_SUMMARY.md
│       ├── INTEGRATION_CHANGELOG.md
│       ├── BACKEND_INTEGRATION_COMPLETE.md
│       ├── FRONTEND_BACKEND_INTEGRATION_GUIDE.md
│       └── RESTORATION_COMPLETE.md
│
├── negotiation_chatbot/               # Main application
│   ├── __pycache__/
│   ├── *.py (all chatbot modules)
│
├── backend/                           # Backend API
│   └── app/
│       ├── *.py (API routes, models)
│
├── resource-hub-main/                 # Frontend React app
│   ├── src/
│   ├── public/
│   └── package.json
│
├── deal_or_no_dialog/                 # Dataset
│   └── exported/
│       ├── train.jsonl
│       ├── validation.jsonl
│       └── test.jsonl
│
├── data/                              # Application data
│   ├── casino.json
│   └── (conversation JSON files)
│
├── chroma_db/                         # Vector database
├── cache/                             # Cache files (gitignored)
└── .venv/                             # Virtual environment
```

## Execution Steps

1. Create scripts/ directory
2. Move utility scripts to scripts/
3. Create docs/archive/ directory
4. Move archived docs to docs/archive/
5. Remove redundant root documentation
6. Rename START_HERE_NEW.md to START_HERE.md
7. Clean cache directory
8. Clean data directory of temp files
9. Update .gitignore
10. Create final README with clear structure
