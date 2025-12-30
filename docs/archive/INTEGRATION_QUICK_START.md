# 🚀 Resource Management System - Quick Start

## One-Minute Setup

### 1. Start Backend (Terminal 1)
```bash
cd /Users/yeonjune.kim.27/Desktop/chatbot/backend
./start_backend.sh
```
✅ Backend running at: http://localhost:8000

### 2. Start Frontend (Terminal 2)
```bash
cd /Users/yeonjune.kim.27/Desktop/chatbot/resource-hub-main
npm run dev
```
✅ Frontend running at: http://localhost:8080

### 3. Login
Open http://localhost:8080 and login with:
- **Email:** `admin@company.com`
- **Password:** `admin`

---

## 📍 Important URLs

| Service | URL | Purpose |
|---------|-----|---------|
| **Frontend UI** | http://localhost:8080 | Main application interface |
| **Backend API** | http://localhost:8000 | REST API server |
| **API Docs** | http://localhost:8000/docs | Interactive API documentation |
| **Health Check** | http://localhost:8000/health | Backend status |

---

## 🔑 Test Credentials

| Email | Password | Role |
|-------|----------|------|
| admin@company.com | admin | Admin |
| user@company.com | user | User |
| john@company.com | password | User |

---

## 📡 API Endpoints Summary

### Authentication
- `POST /api/auth/login` - User login
- `POST /api/auth/logout` - User logout

### Dashboard
- `GET /api/dashboard/stats` - Get statistics

### Resources (CRUD for each)
- `GET /api/employees` - List all
- `POST /api/employees` - Create new
- `PUT /api/employees/{id}` - Update
- `DELETE /api/employees/{id}` - Delete

**Same pattern for:**
- `/api/equipment`
- `/api/inventory`
- `/api/rooms`
- `/api/bookings`

---

## 🧪 Quick Test

### Test Backend API
```bash
# Get dashboard stats
curl http://localhost:8000/api/dashboard/stats

# Get all employees
curl http://localhost:8000/api/employees

# Login
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"admin@company.com","password":"admin"}'
```

### Test Frontend
1. Login at http://localhost:8080
2. Click "Dashboard" - see statistics
3. Click "Employees" - view employee list
4. Try creating a new employee
5. Try editing/deleting

---

## 📁 Project Structure

```
chatbot/
├── backend/                    # Python FastAPI Backend
│   ├── app/
│   │   ├── main.py             # Main server
│   │   ├── frontend_routes.py  # Resource API endpoints
│   │   ├── auth_routes.py      # Authentication endpoints
│   │   ├── frontend_storage.py # In-memory storage
│   │   └── models.py           # Data models
│   └── start_backend.sh        # Startup script
│
└── resource-hub-main/          # React Frontend
    ├── src/
    │   ├── pages/              # UI pages
    │   ├── services/
    │   │   └── api.ts          # API service layer
    │   └── types/
    │       └── resources.ts    # TypeScript interfaces
    └── .env                    # Environment config
```

---

## 🔧 Configuration Files

### Backend
No configuration needed - works out of the box!

### Frontend (.env)
```bash
VITE_API_URL=http://localhost:8000
```

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| Backend won't start | Check if port 8000 is available: `lsof -ti :8000 \| xargs kill -9` |
| Frontend won't start | Check if port 8080 is available, run `npm install` |
| Can't connect | Ensure both frontend and backend are running |
| Login fails | Use exact credentials: `admin@company.com` / `admin` |
| Data resets | Backend uses in-memory storage - restarts clear data |

---

## 📚 Documentation

**For detailed information, see:**
- [FRONTEND_BACKEND_INTEGRATION_GUIDE.md](./FRONTEND_BACKEND_INTEGRATION_GUIDE.md) - Complete integration guide
- [backend/FRONTEND_INTEGRATION_GUIDE.md](./backend/FRONTEND_INTEGRATION_GUIDE.md) - Backend API documentation

---

## ✨ Features Available

✅ User authentication (login/logout)
✅ Dashboard with live statistics
✅ Employee management (CRUD)
✅ Equipment tracking (CRUD)
✅ Inventory management (CRUD)
✅ Room management (CRUD)
✅ Booking system (CRUD)
✅ Sample data pre-loaded
✅ Responsive UI with modern design
✅ Real-time error handling
✅ Type-safe API integration

---

## 🎯 What You Can Do

### Dashboard
- View total employees, equipment, rooms
- See active employees and available equipment
- Monitor low stock items
- Track today's bookings

### Employees
- Add new employees
- Update employee information
- Change employee status (active/inactive/on-leave)
- Delete employees

### Equipment
- Track all equipment with serial numbers
- Assign equipment to employees
- Update status (available/in-use/maintenance/retired)
- Manage equipment locations

### Inventory
- Monitor stock levels
- Set reorder levels
- View low stock alerts
- Manage inventory across locations

### Rooms
- Manage meeting rooms and spaces
- Set room capacity
- List amenities
- Update availability status

### Bookings
- Book rooms and equipment
- View all reservations
- Update booking status
- Cancel bookings

---

**Last Updated:** 2025-12-29
**Status:** ✅ Ready to Use
