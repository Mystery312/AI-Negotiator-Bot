// Resource Management System - Type Definitions
// These types define the structure of all resources in the system

export type UserRole = 'admin' | 'user';

export interface User {
  id: string;
  email: string;
  name: string;
  role: UserRole;
}

export interface Employee {
  id: string;
  name: string;
  email: string;
  department: string;
  position: string;
  status: 'active' | 'inactive' | 'on-leave';
  hireDate: string;
}

export interface Equipment {
  id: string;
  name: string;
  category: string;
  status: 'available' | 'in-use' | 'maintenance' | 'retired';
  assignedTo: string | null;
  location: string;
  serialNumber: string;
}

export interface InventoryItem {
  id: string;
  name: string;
  category: string;
  quantity: number;
  reorderLevel: number;
  location: string;
  unit: string;
}

export interface Room {
  id: string;
  name: string;
  capacity: number;
  floor: string;
  amenities: string[];
  status: 'available' | 'occupied' | 'maintenance';
}

export interface Booking {
  id: string;
  resourceType: 'room' | 'equipment';
  resourceId: string;
  resourceName: string;
  bookedBy: string;
  startTime: string;
  endTime: string;
  purpose: string;
  status: 'confirmed' | 'pending' | 'cancelled';
}

// Dashboard statistics type
export interface DashboardStats {
  totalEmployees: number;
  activeEmployees: number;
  totalEquipment: number;
  availableEquipment: number;
  lowStockItems: number;
  totalRooms: number;
  roomsInUse: number;
  todayBookings: number;
}
