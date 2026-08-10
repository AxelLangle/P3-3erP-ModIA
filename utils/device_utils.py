# device_utils.py
import torch

# Detecta el mejor dispositivo disponible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_device():
    """Devuelve el dispositivo óptimo (GPU si está disponible, CPU si no)."""
    return device

def prepare_for_device(model, *tensors):
    """
    Mueve el modelo y los tensores al dispositivo correcto.
    Retorna el modelo y los tensores en el mismo orden.
    """
    model = model.to(device)
    tensors = [t.to(device) for t in tensors]
    return (model, *tensors)