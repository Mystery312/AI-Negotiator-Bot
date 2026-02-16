// Resource Management System - API Service Layer
// Connected to Python FastAPI Backend on http://localhost:8000

import type {
  Employee,
  Equipment,
  InventoryItem,
  Room,
  Booking,
  DashboardStats,
  User,
} from '@/types/resources';

// Get API base URL from environment variable
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// ============================================
// HELPER FUNCTIONS
// ============================================

async function apiRequest<T>(
  endpoint: string,
  options?: RequestInit
): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;

  try {
    const response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers,
      },
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('API Request Error:', error);
    throw error;
  }
}

// ============================================
// AUTHENTICATION API
// ============================================

export async function loginUser(email: string, password: string): Promise<User | null> {
  try {
    const response = await apiRequest<{ user: User; token?: string }>('/api/auth/login', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    });

    // Store token if provided (for future JWT implementation)
    if (response.token) {
      localStorage.setItem('authToken', response.token);
    }

    return response.user;
  } catch (error) {
    console.error('Login error:', error);
    return null;
  }
}

export async function logoutUser(): Promise<void> {
  try {
    await apiRequest('/api/auth/logout', { method: 'POST' });
    // Clear stored token
    localStorage.removeItem('authToken');
  } catch (error) {
    console.error('Logout error:', error);
    // Clear token even if logout fails
    localStorage.removeItem('authToken');
  }
}

// ============================================
// DASHBOARD API
// ============================================

export async function getDashboardStats(): Promise<DashboardStats> {
  return await apiRequest<DashboardStats>('/api/dashboard/stats');
}

// ============================================
// EMPLOYEES API
// ============================================

export async function getEmployees(): Promise<Employee[]> {
  return await apiRequest<Employee[]>('/api/employees');
}

export async function createEmployee(employee: Omit<Employee, 'id'>): Promise<Employee> {
  return await apiRequest<Employee>('/api/employees', {
    method: 'POST',
    body: JSON.stringify(employee),
  });
}

export async function updateEmployee(id: string, employee: Partial<Employee>): Promise<Employee> {
  return await apiRequest<Employee>(`/api/employees/${id}`, {
    method: 'PUT',
    body: JSON.stringify(employee),
  });
}

export async function deleteEmployee(id: string): Promise<void> {
  await apiRequest<{ message: string }>(`/api/employees/${id}`, {
    method: 'DELETE',
  });
}

// ============================================
// EQUIPMENT API
// ============================================

export async function getEquipment(): Promise<Equipment[]> {
  return await apiRequest<Equipment[]>('/api/equipment');
}

export async function createEquipment(equipment: Omit<Equipment, 'id'>): Promise<Equipment> {
  return await apiRequest<Equipment>('/api/equipment', {
    method: 'POST',
    body: JSON.stringify(equipment),
  });
}

export async function updateEquipment(id: string, equipment: Partial<Equipment>): Promise<Equipment> {
  return await apiRequest<Equipment>(`/api/equipment/${id}`, {
    method: 'PUT',
    body: JSON.stringify(equipment),
  });
}

export async function deleteEquipment(id: string): Promise<void> {
  await apiRequest<{ message: string }>(`/api/equipment/${id}`, {
    method: 'DELETE',
  });
}

// ============================================
// INVENTORY API
// ============================================

export async function getInventory(): Promise<InventoryItem[]> {
  return await apiRequest<InventoryItem[]>('/api/inventory');
}

export async function createInventoryItem(item: Omit<InventoryItem, 'id'>): Promise<InventoryItem> {
  return await apiRequest<InventoryItem>('/api/inventory', {
    method: 'POST',
    body: JSON.stringify(item),
  });
}

export async function updateInventoryItem(id: string, item: Partial<InventoryItem>): Promise<InventoryItem> {
  return await apiRequest<InventoryItem>(`/api/inventory/${id}`, {
    method: 'PUT',
    body: JSON.stringify(item),
  });
}

export async function deleteInventoryItem(id: string): Promise<void> {
  await apiRequest<{ message: string }>(`/api/inventory/${id}`, {
    method: 'DELETE',
  });
}

// ============================================
// ROOMS API
// ============================================

export async function getRooms(): Promise<Room[]> {
  return await apiRequest<Room[]>('/api/rooms');
}

export async function createRoom(room: Omit<Room, 'id'>): Promise<Room> {
  return await apiRequest<Room>('/api/rooms', {
    method: 'POST',
    body: JSON.stringify(room),
  });
}

export async function updateRoom(id: string, room: Partial<Room>): Promise<Room> {
  return await apiRequest<Room>(`/api/rooms/${id}`, {
    method: 'PUT',
    body: JSON.stringify(room),
  });
}

export async function deleteRoom(id: string): Promise<void> {
  await apiRequest<{ message: string }>(`/api/rooms/${id}`, {
    method: 'DELETE',
  });
}

// ============================================
// BOOKINGS API
// ============================================

export async function getBookings(): Promise<Booking[]> {
  return await apiRequest<Booking[]>('/api/bookings');
}

export async function createBooking(booking: Omit<Booking, 'id'>): Promise<Booking> {
  return await apiRequest<Booking>('/api/bookings', {
    method: 'POST',
    body: JSON.stringify(booking),
  });
}

export async function updateBooking(id: string, booking: Partial<Booking>): Promise<Booking> {
  return await apiRequest<Booking>(`/api/bookings/${id}`, {
    method: 'PUT',
    body: JSON.stringify(booking),
  });
}

export async function deleteBooking(id: string): Promise<void> {
  await apiRequest<{ message: string }>(`/api/bookings/${id}`, {
    method: 'DELETE',
  });
}
