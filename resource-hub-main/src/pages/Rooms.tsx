import { useEffect, useState } from 'react';
import { AppLayout } from '@/components/layout/AppLayout';
import { DataTable, StatusBadge } from '@/components/DataTable';
import { getRooms, createRoom, updateRoom, deleteRoom } from '@/services/api';
import type { Room } from '@/types/resources';
import { useToast } from '@/hooks/use-toast';
import { Badge } from '@/components/ui/badge';

export default function Rooms() {
  const [rooms, setRooms] = useState<Room[]>([]);
  const [loading, setLoading] = useState(true);
  const { toast } = useToast();

  useEffect(() => {
    loadRooms();
  }, []);

  const loadRooms = async () => {
    const data = await getRooms();
    setRooms(data);
    setLoading(false);
  };

  const handleAdd = async (room: Omit<Room, 'id'>) => {
    // Parse amenities from comma-separated string if needed
    const processedRoom = {
      ...room,
      amenities: typeof room.amenities === 'string' 
        ? (room.amenities as string).split(',').map((a) => a.trim())
        : room.amenities,
    };
    const newRoom = await createRoom(processedRoom);
    setRooms((prev) => [...prev, newRoom]);
    toast({ title: 'Room added', description: 'New room has been added.' });
  };

  const handleEdit = async (id: string, updates: Partial<Room>) => {
    // Parse amenities from comma-separated string if needed
    const processedUpdates = {
      ...updates,
      amenities: typeof updates.amenities === 'string'
        ? (updates.amenities as string).split(',').map((a) => a.trim())
        : updates.amenities,
    };
    await updateRoom(id, processedUpdates);
    setRooms((prev) =>
      prev.map((room) => (room.id === id ? { ...room, ...processedUpdates } : room))
    );
    toast({ title: 'Room updated', description: 'Room details have been updated.' });
  };

  const handleDelete = async (id: string) => {
    await deleteRoom(id);
    setRooms((prev) => prev.filter((room) => room.id !== id));
    toast({ title: 'Room deleted', description: 'Room has been removed.' });
  };

  if (loading) {
    return (
      <AppLayout>
        <div className="flex items-center justify-center h-64">
          <p className="text-muted-foreground">Loading...</p>
        </div>
      </AppLayout>
    );
  }

  return (
    <AppLayout>
      <div className="space-y-6">
        <div>
          <h1 className="text-2xl font-semibold text-foreground">Rooms</h1>
          <p className="text-muted-foreground">Manage meeting spaces and facilities</p>
        </div>

        <DataTable
          data={rooms}
          title="Rooms"
          searchKeys={['name', 'floor']}
          columns={[
            { key: 'name', label: 'Name' },
            { key: 'capacity', label: 'Capacity', render: (r) => `${r.capacity} people` },
            { key: 'floor', label: 'Floor' },
            {
              key: 'amenities',
              label: 'Amenities',
              render: (r) => (
                <div className="flex flex-wrap gap-1">
                  {r.amenities.slice(0, 2).map((a) => (
                    <Badge key={a} variant="outline" className="text-xs">
                      {a}
                    </Badge>
                  ))}
                  {r.amenities.length > 2 && (
                    <Badge variant="outline" className="text-xs">
                      +{r.amenities.length - 2}
                    </Badge>
                  )}
                </div>
              ),
            },
            {
              key: 'status',
              label: 'Status',
              render: (r) => <StatusBadge status={r.status} />,
            },
          ]}
          formFields={[
            { key: 'name', label: 'Name', type: 'text', required: true },
            { key: 'capacity', label: 'Capacity', type: 'number', required: true },
            { key: 'floor', label: 'Floor', type: 'text', required: true },
            { key: 'amenities', label: 'Amenities (comma-separated)', type: 'text' },
            {
              key: 'status',
              label: 'Status',
              type: 'select',
              options: [
                { value: 'available', label: 'Available' },
                { value: 'occupied', label: 'Occupied' },
                { value: 'maintenance', label: 'Maintenance' },
              ],
            },
          ]}
          onAdd={handleAdd}
          onEdit={handleEdit}
          onDelete={handleDelete}
        />
      </div>
    </AppLayout>
  );
}
