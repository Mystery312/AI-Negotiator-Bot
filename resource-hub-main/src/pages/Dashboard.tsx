import { useEffect, useState } from 'react';
import { AppLayout } from '@/components/layout/AppLayout';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { getDashboardStats } from '@/services/api';
import type { DashboardStats } from '@/types/resources';
import { Users, Package, Warehouse, DoorOpen, AlertTriangle, Calendar } from 'lucide-react';

export default function Dashboard() {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getDashboardStats().then((data) => {
      setStats(data);
      setLoading(false);
    });
  }, []);

  if (loading) {
    return (
      <AppLayout>
        <div className="flex items-center justify-center h-64">
          <p className="text-muted-foreground">Loading...</p>
        </div>
      </AppLayout>
    );
  }

  const statCards = [
    {
      title: 'Total Employees',
      value: stats?.totalEmployees || 0,
      subtitle: `${stats?.activeEmployees || 0} active`,
      icon: <Users className="h-5 w-5 text-muted-foreground" />,
    },
    {
      title: 'Equipment',
      value: stats?.totalEquipment || 0,
      subtitle: `${stats?.availableEquipment || 0} available`,
      icon: <Package className="h-5 w-5 text-muted-foreground" />,
    },
    {
      title: 'Low Stock Items',
      value: stats?.lowStockItems || 0,
      subtitle: 'Need reorder',
      icon: <AlertTriangle className="h-5 w-5 text-warning" />,
      highlight: (stats?.lowStockItems || 0) > 0,
    },
    {
      title: 'Rooms',
      value: stats?.totalRooms || 0,
      subtitle: `${stats?.roomsInUse || 0} in use`,
      icon: <DoorOpen className="h-5 w-5 text-muted-foreground" />,
    },
    {
      title: "Today's Bookings",
      value: stats?.todayBookings || 0,
      subtitle: 'Scheduled',
      icon: <Calendar className="h-5 w-5 text-muted-foreground" />,
    },
  ];

  return (
    <AppLayout>
      <div className="space-y-6">
        <div>
          <h1 className="text-2xl font-semibold text-foreground">Dashboard</h1>
          <p className="text-muted-foreground">Overview of your resources</p>
        </div>

        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5">
          {statCards.map((card) => (
            <Card key={card.title}>
              <CardHeader className="flex flex-row items-center justify-between pb-2">
                <CardTitle className="text-sm font-medium text-muted-foreground">
                  {card.title}
                </CardTitle>
                {card.icon}
              </CardHeader>
              <CardContent>
                <div className={`text-2xl font-semibold ${card.highlight ? 'text-warning' : ''}`}>
                  {card.value}
                </div>
                <p className="text-xs text-muted-foreground mt-1">{card.subtitle}</p>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Placeholder for additional dashboard content */}
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Quick Actions</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">
              Use the sidebar to navigate between different resource types. You can add, edit, and delete resources as needed.
            </p>
          </CardContent>
        </Card>
      </div>
    </AppLayout>
  );
}
