import { useEffect, useState } from 'react';
import { AppLayout } from '@/components/layout/AppLayout';
import { DataTable, StatusBadge } from '@/components/DataTable';
import { getBookings, createBooking, deleteBooking } from '@/services/api';
import type { Booking } from '@/types/resources';
import { useToast } from '@/hooks/use-toast';

export default function Bookings() {
  const [bookings, setBookings] = useState<Booking[]>([]);
  const [loading, setLoading] = useState(true);
  const { toast } = useToast();

  useEffect(() => {
    getBookings().then((data) => {
      setBookings(data);
      setLoading(false);
    });
  }, []);

  const handleAdd = async (booking: Omit<Booking, 'id'>) => {
    const newBooking = await createBooking(booking);
    setBookings((prev) => [...prev, newBooking]);
    toast({ title: 'Booking created', description: 'New booking has been added.' });
  };

  const handleDelete = async (id: string) => {
    await deleteBooking(id);
    setBookings((prev) => prev.filter((b) => b.id !== id));
    toast({ title: 'Booking cancelled', description: 'Booking has been removed.' });
  };

  if (loading) {
    return <AppLayout><div className="flex items-center justify-center h-64"><p className="text-muted-foreground">Loading...</p></div></AppLayout>;
  }

  return (
    <AppLayout>
      <div className="space-y-6">
        <div>
          <h1 className="text-2xl font-semibold text-foreground">Bookings</h1>
          <p className="text-muted-foreground">Manage room and equipment reservations</p>
        </div>
        <DataTable
          data={bookings}
          title="Bookings"
          searchKeys={['resourceName', 'bookedBy', 'purpose']}
          columns={[
            { key: 'resourceType', label: 'Type', render: (b) => <span className="capitalize">{b.resourceType}</span> },
            { key: 'resourceName', label: 'Resource' },
            { key: 'bookedBy', label: 'Booked By' },
            { key: 'startTime', label: 'Start' },
            { key: 'endTime', label: 'End' },
            { key: 'purpose', label: 'Purpose' },
            { key: 'status', label: 'Status', render: (b) => <StatusBadge status={b.status} /> },
          ]}
          formFields={[
            { key: 'resourceType', label: 'Type', type: 'select', options: [{ value: 'room', label: 'Room' }, { value: 'equipment', label: 'Equipment' }] },
            { key: 'resourceName', label: 'Resource Name', type: 'text', required: true },
            { key: 'bookedBy', label: 'Booked By', type: 'text', required: true },
            { key: 'startTime', label: 'Start Time', type: 'text', required: true },
            { key: 'endTime', label: 'End Time', type: 'text', required: true },
            { key: 'purpose', label: 'Purpose', type: 'text', required: true },
            { key: 'status', label: 'Status', type: 'select', options: [{ value: 'pending', label: 'Pending' }, { value: 'confirmed', label: 'Confirmed' }] },
          ]}
          onAdd={handleAdd}
          onDelete={handleDelete}
        />
      </div>
    </AppLayout>
  );
}
