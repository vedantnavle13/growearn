import { useState } from 'react';
import type { Address } from '../types';

interface AddressSelectorProps {
  addresses: Address[];
  onSelect: (address: Address) => void;
  onAddAddress: (address: Omit<Address, 'id' | 'isDefault'>) => void;
}

export function AddressSelector({ addresses, onSelect, onAddAddress }: AddressSelectorProps) {
  const [showAddForm, setShowAddForm] = useState(false);
  const [formData, setFormData] = useState({
    label: '',
    recipientName: '',
    addressLine1: '',
    addressLine2: '',
    city: '',
    state: '',
    postalCode: '',
    country: 'India',
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onAddAddress(formData);
    setShowAddForm(false);
    setFormData({
      label: '',
      recipientName: '',
      addressLine1: '',
      addressLine2: '',
      city: '',
      state: '',
      postalCode: '',
      country: 'India',
    });
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>) => {
    setFormData((prev) => ({ ...prev, [e.target.name]: e.target.value }));
  };

  return (
    <div className="address-selector">
      <h4>Delivery Address</h4>

      {addresses.length > 0 && (
        <div className="saved-addresses">
          {addresses.map((address) => (
            <label key={address.id} className="address-option">
              <input
                type="radio"
                name="address"
                value={address.id}
                onChange={() => onSelect(address)}
              />
              <div className="address-details">
                <div className="address-label">
                  {address.label} {address.isDefault && <span className="default-badge">Default</span>}
                </div>
                <div className="address-text">
                  {address.recipientName}<br />
                  {address.addressLine1}
                  {address.addressLine2 && `, ${address.addressLine2}`}<br />
                  {address.city}, {address.state} {address.postalCode}<br />
                  {address.country}
                </div>
              </div>
            </label>
          ))}
        </div>
      )}

      {addresses.length === 0 || showAddForm ? (
        <form onSubmit={handleSubmit} className="address-form">
          <h5>{addresses.length === 0 ? 'Add Delivery Address' : 'Add New Address'}</h5>
          
          <div className="form-row">
            <div className="form-group">
              <label htmlFor="label">Label (e.g., Home, Office)</label>
              <input
                type="text"
                id="label"
                name="label"
                value={formData.label}
                onChange={handleChange}
                required
              />
            </div>
            <div className="form-group">
              <label htmlFor="recipientName">Recipient Name</label>
              <input
                type="text"
                id="recipientName"
                name="recipientName"
                value={formData.recipientName}
                onChange={handleChange}
                required
              />
            </div>
          </div>

          <div className="form-group">
            <label htmlFor="addressLine1">Address Line 1</label>
            <input
              type="text"
              id="addressLine1"
              name="addressLine1"
              value={formData.addressLine1}
              onChange={handleChange}
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="addressLine2">Address Line 2 (optional)</label>
            <input
              type="text"
              id="addressLine2"
              name="addressLine2"
              value={formData.addressLine2}
              onChange={handleChange}
            />
          </div>

          <div className="form-row">
            <div className="form-group">
              <label htmlFor="city">City</label>
              <input
                type="text"
                id="city"
                name="city"
                value={formData.city}
                onChange={handleChange}
                required
              />
            </div>
            <div className="form-group">
              <label htmlFor="state">State</label>
              <input
                type="text"
                id="state"
                name="state"
                value={formData.state}
                onChange={handleChange}
                required
              />
            </div>
            <div className="form-group">
              <label htmlFor="postalCode">Postal Code</label>
              <input
                type="text"
                id="postalCode"
                name="postalCode"
                value={formData.postalCode}
                onChange={handleChange}
                required
              />
            </div>
          </div>

          <div className="form-group">
            <label htmlFor="country">Country</label>
            <select
              id="country"
              name="country"
              value={formData.country}
              onChange={handleChange}
              required
            >
              <option value="India">India</option>
            </select>
          </div>

          <div className="form-actions">
            <button type="submit" className="btn-save-address">
              Save Address
            </button>
            {!addresses.length && (
              <button type="button" className="btn-cancel" onClick={() => setShowAddForm(false)}>
                Cancel
              </button>
            )}
          </div>
        </form>
      ) : (
        <button className="btn-add-address" onClick={() => setShowAddForm(true)}>
          + Add New Address
        </button>
      )}
    </div>
  );
}