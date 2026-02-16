import { useEffect, useState } from 'react';
import { AppLayout } from '@/components/layout/AppLayout';
import { DataTable, StatusBadge } from '@/components/DataTable';
import { getEquipment, createEquipment, updateEquipment, deleteEquipment } from '@/services/api';
import type { Equipment } from '@/types/resources';
import { useToast } from '@/hooks/use-toast';

export default function EquipmentPage() {
  const [equipment, setEquipment] = useState<Equipment[]>([]);
  const [loading, setLoading] = useState(true);
  const { toast } = useToast();

  useEffect(() => {
    loadEquipment();
  }, []);

  const loadEquipment = async () => {
    const data = await getEquipment();
    setEquipment(data);
    setLoading(false);
  };

  const handleAdd = async (item: Omit<Equipment, 'id'>) => {
    const newItem = await createEquipment(item);
    setEquipment((prev) => [...prev, newItem]);
    toast({ title: 'Equipment added', description: 'New equipment has been added.' });
  };

  const handleEdit = async (id: string, updates: Partial<Equipment>) => {
    await updateEquipment(id, updates);
    setEquipment((prev) =>
      prev.map((item) => (item.id === id ? { ...item, ...updates } : item))
    );
    toast({ title: 'Equipment updated', description: 'Equipment details have been updated.' });
  };

  const handleDelete = async (id: string) => {
    await deleteEquipment(id);
    setEquipment((prev) => prev.filter((item) => item.id !== id));
    toast({ title: 'Equipment deleted', description: 'Equipment has been removed.' });
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
          <h1 className="text-2xl font-semibold text-foreground">Equipment</h1>
          <p className="text-muted-foreground">Manage assets and devices</p>
        </div>

        <DataTable
          data={equipment}
          title="Equipment"
          searchKeys={['name', 'category', 'serialNumber', 'location']}
          columns={[
            { key: 'name', label: 'Name' },
            { key: 'category', label: 'Category' },
            { key: 'serialNumber', label: 'Serial Number' },
            { key: 'location', label: 'Location' },
            { key: 'assignedTo', label: 'Assigned To', render: (e) => e.assignedTo || '—' },
            {
              key: 'status',
              label: 'Status',
              render: (e) => <StatusBadge status={e.status} />,
            },
          ]}
          formFields={[
            { key: 'name', label: 'Name', type: 'text', required: true },
            { key: 'category', label: 'Category', type: 'text', required: true },
            { key: 'serialNumber', label: 'Serial Number', type: 'text', required: true },
            { key: 'location', label: 'Location', type: 'text', required: true },
            { key: 'assignedTo', label: 'Assigned To', type: 'text' },
            {
              key: 'status',
              label: 'Status',
              type: 'select',
              options: [
                { value: 'available', label: 'Available' },
                { value: 'in-use', label: 'In Use' },
                { value: 'maintenance', label: 'Maintenance' },
                { value: 'retired', label: 'Retired' },
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
