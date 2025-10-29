import numpy as np
import xgboost as xgb
import torch

class BDTVertexFinder:
    """
    Boosted Decision Tree para clasificación de pares de trazas.
    Compatible con la interfaz de SetToGraph.
    """
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1):
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        )
        self.trained = False
        self.device = 'cpu'
        # Acumular datos de todos los batches antes de entrenar
        self.X_accumulated = []
        self.y_accumulated = []
        self.accumulation_mode = True
    
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
        
        if not self.trained:
            return edge_vals
        
        for b in range(B):
            # Crear features para cada par de trazas
            pair_features = []
            for i in range(N):
                for j in range(N):
                    if i != j:  # Excluir diagonal
                        # Concatenar features de ambas trazas
                        feat_ij = np.concatenate([
                            x[b, i].cpu().numpy(),
                            x[b, j].cpu().numpy()
                        ])
                        pair_features.append(feat_ij)
            
            if len(pair_features) > 0:
                pair_features = np.array(pair_features)
                # Predecir probabilidades (logits para compatibilidad)
                probs = self.model.predict_proba(pair_features)[:, 1]
                # Convertir a logits
                logits = np.log(probs + 1e-10) - np.log(1 - probs + 1e-10)
                
                # Convertir a tensor de torch
                logits_tensor = torch.from_numpy(logits).to(x.device)
                
                # Reconstruir matriz de adyacencia
                idx = 0
                for i in range(N):
                    for j in range(N):
                        if i != j:
                            edge_vals[b, i, j] = logits_tensor[idx]
                            idx += 1
        
        return edge_vals.unsqueeze(1)  # Shape (B, 1, N, N) para compatibilidad
    
    def __call__(self, x):
        """Make the model callable"""
        return self.forward(x)
    
    def fit_batch(self, x, y):
        """
        Entrena el BDT con un batch de datos.
        
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
                        X_train.append(feat_ij)
                        y_train.append(int(y[b, i, j].item()))
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Acumular datos
        self.X_accumulated.append(X_train)
        self.y_accumulated.append(y_train)
    
    def train_model(self):
        """
        Entrena el modelo con todos los datos acumulados.
        Llamar al final de cada época.
        """
        if len(self.X_accumulated) == 0:
            return
        
        # Concatenar todos los datos acumulados
        X_all = np.vstack(self.X_accumulated)
        y_all = np.concatenate(self.y_accumulated)
        
        # Entrenar
        if not self.trained:
            self.model.fit(X_all, y_all)
            self.trained = True
        else:
            # Re-entrenar con todos los datos (BDT no soporta fit incremental bien)
            self.model.fit(X_all, y_all)
        
        # Limpiar acumuladores
        self.X_accumulated = []
        self.y_accumulated = []
    
    def parameters(self):
        """Compatibilidad con optimizadores de PyTorch"""
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
        """Guardar estado del modelo"""
        import pickle
        return {
            'model': pickle.dumps(self.model),
            'trained': self.trained
        }
    
    def load_state_dict(self, state_dict):
        """Cargar estado del modelo"""
        import pickle
        self.model = pickle.loads(state_dict['model'])
        self.trained = state_dict['trained']
