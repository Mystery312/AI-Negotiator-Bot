import { useEffect, useState } from 'react';
import { AppLayout } from '@/components/layout/AppLayout';
import { DataTable, StatusBadge } from '@/components/DataTable';
import { getEmployees, createEmployee, updateEmployee, deleteEmployee } from '@/services/api';
import type { Employee } from '@/types/resources';
import { useToast } from '@/hooks/use-toast';

export default function Employees() {
  const [employees, setEmployees] = useState<Employee[]>([]);
  const [loading, setLoading] = useState(true);
  const { toast } = useToast();

  useEffect(() => {
    loadEmployees();
  }, []);

  const loadEmployees = async () => {
    const data = await getEmployees();
    setEmployees(data);
    setLoading(false);
  };

  const handleAdd = async (employee: Omit<Employee, 'id'>) => {
    const newEmployee = await createEmployee(employee);
    setEmployees((prev) => [...prev, newEmployee]);
    toast({ title: 'Employee added', description: 'New employee has been added successfully.' });
  };

  const handleEdit = async (id: string, updates: Partial<Employee>) => {
    await updateEmployee(id, updates);
    setEmployees((prev) =>
      prev.map((emp) => (emp.id === id ? { ...emp, ...updates } : emp))
    );
    toast({ title: 'Employee updated', description: 'Employee details have been updated.' });
  };

  const handleDelete = async (id: string) => {
    await deleteEmployee(id);
    setEmployees((prev) => prev.filter((emp) => emp.id !== id));
    toast({ title: 'Employee deleted', description: 'Employee has been removed.' });
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
          <h1 className="text-2xl font-semibold text-foreground">Employees</h1>
          <p className="text-muted-foreground">Manage staff and personnel</p>
        </div>

        <DataTable
          data={employees}
          title="Employees"
          searchKeys={['name', 'email', 'department', 'position']}
          columns={[
            { key: 'name', label: 'Name' },
            { key: 'email', label: 'Email' },
            { key: 'department', label: 'Department' },
            { key: 'position', label: 'Position' },
            {
              key: 'status',
              label: 'Status',
              render: (emp) => <StatusBadge status={emp.status} />,
            },
          ]}
          formFields={[
            { key: 'name', label: 'Name', type: 'text', required: true },
            { key: 'email', label: 'Email', type: 'email', required: true },
            { key: 'department', label: 'Department', type: 'text', required: true },
            { key: 'position', label: 'Position', type: 'text', required: true },
            {
              key: 'status',
              label: 'Status',
              type: 'select',
              options: [
                { value: 'active', label: 'Active' },
                { value: 'inactive', label: 'Inactive' },
                { value: 'on-leave', label: 'On Leave' },
              ],
            },
            { key: 'hireDate', label: 'Hire Date', type: 'text', required: true },
          ]}
          onAdd={handleAdd}
          onEdit={handleEdit}
          onDelete={handleDelete}
        />
      </div>
    </AppLayout>
  );
}
