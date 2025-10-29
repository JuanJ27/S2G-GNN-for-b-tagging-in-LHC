import numpy as np
import torch
import pennylane as qml
from pennylane import numpy as pnp

class QuantumAngleEmbedding:
    """
    Quantum Neural Network usando Angle Embedding con PennyLane.
    Compatible con la interfaz de SetToGraph.
    """
    def __init__(self, n_qubits=10, n_layers=3, learning_rate=0.01):
        """
        Args:
            n_qubits: Número de qubits (debe coincidir con dimensión de features)
            n_layers: Número de capas del circuito variacional
            learning_rate: Learning rate para optimización
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.device = 'cpu'
        
        # Crear dispositivo cuántico (simulador)
        self.dev = qml.device('default.qubit', wires=n_qubits)
        
        # Inicializar parámetros del circuito variacional
        # Shape: (n_layers, n_qubits, 3) para rotaciones Rot(phi, theta, omega)
        self.params = pnp.random.uniform(0, 2*np.pi, (n_layers, n_qubits, 3), requires_grad=True)
        
        # Crear QNode con interface 'autograd' para compatibilidad con optimizer de PennyLane
        self.qnode = qml.QNode(self._quantum_circuit, self.dev, interface='autograd')
        
        self.trained_epochs = 0
        
        # Acumuladores para batch training
        self.X_accumulated = []
        self.y_accumulated = []
    
    def _quantum_circuit(self, features, params):
        """
        Circuito cuántico con Angle Embedding y capas variacionales.
        
        Args:
            features: Features normalizadas [n_qubits]
            params: Parámetros del circuito [n_layers, n_qubits, 3]
        
        Returns:
            expectation: Valor esperado de medición
        """
        # Angle Embedding: codificar features clásicas en ángulos de rotación
        qml.AngleEmbedding(features, wires=range(self.n_qubits), rotation='Y')
        
        # Capas variacionales
        for layer in range(self.n_layers):
            # Rotaciones parametrizadas
            for wire in range(self.n_qubits):
                qml.Rot(params[layer, wire, 0], 
                       params[layer, wire, 1], 
                       params[layer, wire, 2], 
                       wires=wire)
            
            # Entanglement con CNOT gates
            for wire in range(self.n_qubits - 1):
                qml.CNOT(wires=[wire, wire + 1])
            # Cerrar el anillo
            if self.n_qubits > 1:
                qml.CNOT(wires=[self.n_qubits - 1, 0])
        
        # Medición: valor esperado de PauliZ en el primer qubit
        return qml.expval(qml.PauliZ(0))
    
    def _normalize_features(self, features):
        """
        Normaliza features al rango [0, 2π] para angle embedding.
        
        Args:
            features: numpy array de features
        
        Returns:
            features normalizadas
        """
        # Normalizar cada feature independientemente
        features_min = features.min(axis=0, keepdims=True)
        features_max = features.max(axis=0, keepdims=True)
        features_range = features_max - features_min + 1e-10
        
        normalized = (features - features_min) / features_range * 2 * np.pi
        return normalized
    
    def forward(self, x):
        """
        Interfaz compatible con SetToGraph.
        
        Args:
            x: torch.Tensor de shape (B, N, 10) - features de trazas
        
        Returns:
            edge_vals: torch.Tensor de shape (B, N, N) - scores de aristas
        """
        B, N, F = x.shape
        edge_vals = torch.zeros(B, N, N, device=x.device)
        
        for b in range(B):
            # Crear features para cada par de trazas
            for i in range(N):
                for j in range(N):
                    if i != j:  # Excluir diagonal
                        # Concatenar features de ambas trazas
                        feat_ij = np.concatenate([
                            x[b, i].cpu().numpy(),
                            x[b, j].cpu().numpy()
                        ])
                        
                        # Reducir dimensionalidad si es necesario (20 -> 10)
                        if len(feat_ij) > self.n_qubits:
                            # Tomar las primeras n_qubits features más importantes
                            # o hacer PCA, pero por simplicidad tomamos las primeras
                            feat_ij = feat_ij[:self.n_qubits]
                        elif len(feat_ij) < self.n_qubits:
                            # Pad con ceros si hay menos features
                            feat_ij = np.pad(feat_ij, (0, self.n_qubits - len(feat_ij)))
                        
                        # Normalizar para angle embedding
                        feat_norm = self._normalize_features(feat_ij.reshape(1, -1)).flatten()
                        
                        # Ejecutar circuito cuántico
                        # Usar arrays de numpy (no torch) para compatibilidad con autograd
                        feat_array = pnp.array(feat_norm, requires_grad=False)
                        
                        # Obtener predicción del circuito cuántico
                        # Usar self.params directamente (ya son del tipo correcto)
                        expectation = self.qnode(feat_array, self.params)
                        
                        # Transformar expectation value [-1, 1] a logit [-∞, ∞]
                        # Usar arctanh (inversa de tanh) para mapeo riguroso
                        # Clipear para evitar infinitos en los extremos
                        exp_clipped = np.clip(float(expectation), -0.999, 0.999)
                        logit = np.arctanh(exp_clipped)
                        
                        edge_vals[b, i, j] = logit
        
        return edge_vals.unsqueeze(1)  # Shape (B, 1, N, N) para compatibilidad
    
    def __call__(self, x):
        """Make the model callable"""
        return self.forward(x)
    
    def fit_batch(self, x, y):
        """
        Acumula datos de un batch para entrenamiento.
        
        Args:
            x: torch.Tensor de shape (B, N, 10)
            y: torch.Tensor de shape (B, N, N) - ground truth
        """
        B, N, F = x.shape
        
        # Preparar datos de entrenamiento
        X_train = []
        y_train = []
        
        for b in range(B):
            for i in range(N):
                for j in range(N):
                    if i != j:  # Excluir diagonal
                        feat_ij = np.concatenate([
                            x[b, i].cpu().numpy(),
                            x[b, j].cpu().numpy()
                        ])
                        
                        # Ajustar dimensionalidad
                        if len(feat_ij) > self.n_qubits:
                            feat_ij = feat_ij[:self.n_qubits]
                        elif len(feat_ij) < self.n_qubits:
                            feat_ij = np.pad(feat_ij, (0, self.n_qubits - len(feat_ij)))
                        
                        X_train.append(feat_ij)
                        # Convertir labels binarios: 1 -> 1, 0 -> -1 (para expectation values)
                        y_train.append(2 * int(y[b, i, j].item()) - 1)
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Acumular datos
        self.X_accumulated.append(X_train)
        self.y_accumulated.append(y_train)
    
    def train_model(self):
        """
        Entrena el modelo cuántico con todos los datos acumulados.
        Llamar al final de cada época.
        """
        if len(self.X_accumulated) == 0:
            return
        
        # Concatenar todos los datos acumulados
        X_all = np.vstack(self.X_accumulated)
        y_all = np.concatenate(self.y_accumulated)
        
        # Normalizar features
        X_normalized = self._normalize_features(X_all)
        
        # Optimizer para parámetros cuánticos
        opt = qml.GradientDescentOptimizer(stepsize=self.learning_rate)
        
        # Función de costo
        def cost_fn(params):
            predictions = []
            for features in X_normalized[:100]:  # Limitar por tiempo de ejecución
                # Usar pnp.array para mantener todo en autograd
                feat_array = pnp.array(features, requires_grad=False)
                # params ya está en formato autograd
                pred = self.qnode(feat_array, params)
                predictions.append(pred)
            
            predictions = pnp.array(predictions)  # Usar numpy de PennyLane
            targets = pnp.array(y_all[:100])
            
            # MSE loss entre expectation values y targets
            loss = pnp.mean((predictions - targets) ** 2)
            return loss
        
        # Entrenar por más iteraciones para mejorar aprendizaje
        n_iterations = 20  # Aumentado de 5 a 20 para mejor convergencia
        print(f"  Training QML model for {n_iterations} iterations...")
        
        for i in range(n_iterations):
            self.params = opt.step(cost_fn, self.params)
            
            # Imprimir progreso cada 5 iteraciones
            if i % 5 == 0 or i == n_iterations - 1:
                current_loss = float(cost_fn(self.params))
                print(f"    QML iter {i+1}/{n_iterations}: loss={current_loss:.4f}")
        
        self.trained_epochs += 1
        
        # Limpiar acumuladores
        self.X_accumulated = []
        self.y_accumulated = []
    
    def parameters(self):
        """Retorna parámetros para compatibilidad (aunque no usamos optimizer de PyTorch)"""
        return []
    
    def train(self):
        """Modo entrenamiento"""
        pass
    
    def eval(self):
        """Modo evaluación"""
        pass
    
    def to(self, device):
        """Compatibilidad con device management"""
        self.device = device
        return self
    
    def state_dict(self):
        """Retorna el estado del modelo para guardado"""
        return {
            'params': self.params,
            'n_qubits': self.n_qubits,
            'n_layers': self.n_layers,
            'learning_rate': self.learning_rate,
            'trained_epochs': self.trained_epochs
        }
    
    def load_state_dict(self, state_dict):
        """Carga el estado del modelo"""
        self.params = state_dict['params']
        self.n_qubits = state_dict['n_qubits']
        self.n_layers = state_dict['n_layers']
        self.learning_rate = state_dict['learning_rate']
        self.trained_epochs = state_dict.get('trained_epochs', 0)
        
        # Recrear el dispositivo y QNode con los nuevos parámetros
        self.dev = qml.device('default.qubit', wires=self.n_qubits)
        self.qnode = qml.QNode(self._quantum_circuit, self.dev, interface='autograd')
    
    @classmethod
    def from_state_dict(cls, state_dict):
        """Crea una instancia del modelo desde un state_dict"""
        model = cls(
            n_qubits=state_dict['n_qubits'],
            n_layers=state_dict['n_layers'],
            learning_rate=state_dict['learning_rate']
        )
        model.load_state_dict(state_dict)
        return model
