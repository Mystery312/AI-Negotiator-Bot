import { useEffect, useState } from 'react';
import { AppLayout } from '@/components/layout/AppLayout';
import { DataTable } from '@/components/DataTable';
import { getInventory, createInventoryItem, updateInventoryItem, deleteInventoryItem } from '@/services/api';
import type { InventoryItem } from '@/types/resources';
import { useToast } from '@/hooks/use-toast';
import { Badge } from '@/components/ui/badge';

export default function Inventory() {
  const [inventory, setInventory] = useState<InventoryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const { toast } = useToast();

  useEffect(() => {
    loadInventory();
  }, []);

  const loadInventory = async () => {
    const data = await getInventory();
    setInventory(data);
    setLoading(false);
  };

  const handleAdd = async (item: Omit<InventoryItem, 'id'>) => {
    const newItem = await createInventoryItem(item);
    setInventory((prev) => [...prev, newItem]);
    toast({ title: 'Item added', description: 'New inventory item has been added.' });
  };

  const handleEdit = async (id: string, updates: Partial<InventoryItem>) => {
    await updateInventoryItem(id, updates);
    setInventory((prev) =>
      prev.map((item) => (item.id === id ? { ...item, ...updates } : item))
    );
    toast({ title: 'Item updated', description: 'Inventory item has been updated.' });
  };

  const handleDelete = async (id: string) => {
    await deleteInventoryItem(id);
    setInventory((prev) => prev.filter((item) => item.id !== id));
    toast({ title: 'Item deleted', description: 'Inventory item has been removed.' });
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
          <h1 className="text-2xl font-semibold text-foreground">Inventory</h1>
          <p className="text-muted-foreground">Track supplies and stock levels</p>
        </div>

        <DataTable
          data={inventory}
          title="Items"
          searchKeys={['name', 'category', 'location']}
          columns={[
            { key: 'name', label: 'Name' },
            { key: 'category', label: 'Category' },
            {
              key: 'quantity',
              label: 'Quantity',
              render: (item) => (
                <span className={item.quantity <= item.reorderLevel ? 'text-destructive font-medium' : ''}>
                  {item.quantity} {item.unit}
                </span>
              ),
            },
            { key: 'reorderLevel', label: 'Reorder Level' },
            { key: 'location', label: 'Location' },
            {
              key: 'stockStatus',
              label: 'Status',
              render: (item) =>
                item.quantity <= item.reorderLevel ? (
                  <Badge variant="destructive">Low Stock</Badge>
                ) : (
                  <Badge variant="default">In Stock</Badge>
                ),
            },
          ]}
          formFields={[
            { key: 'name', label: 'Name', type: 'text', required: true },
            { key: 'category', label: 'Category', type: 'text', required: true },
            { key: 'quantity', label: 'Quantity', type: 'number', required: true },
            { key: 'reorderLevel', label: 'Reorder Level', type: 'number', required: true },
            { key: 'unit', label: 'Unit', type: 'text', required: true },
            { key: 'location', label: 'Location', type: 'text', required: true },
          ]}
          onAdd={handleAdd}
          onEdit={handleEdit}
          onDelete={handleDelete}
        />
      </div>
    </AppLayout>
  );
}
