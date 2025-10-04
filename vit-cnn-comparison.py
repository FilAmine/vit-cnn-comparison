import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import timm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm import tqdm
import time

class VisionComparativeStudy:
    def __init__(self, batch_size=128, num_epochs=200):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.results = {}
        
        # Configuration des hyperparamètres
        self.config = {
            'cnn': {
                'lr': 0.1,
                'optimizer': 'SGD',
                'weight_decay': 0.0001,
                'scheduler': 'cosine'
            },
            'vit': {
                'lr': 0.001,
                'optimizer': 'AdamW',
                'weight_decay': 0.05,
                'scheduler': 'cosine',
                'warmup_epochs': 20
            }
        }
    
    def setup_data(self):
        """Configuration des datasets CIFAR-10 et CIFAR-100"""
        print("Setting up CIFAR datasets...")
        
        # Transformations pour l'entraînement
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], 
                              std=[0.2023, 0.1994, 0.2010])
        ])
        
        # Transformations pour le test
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], 
                              std=[0.2023, 0.1994, 0.2010])
        ])
        
        # Chargement des datasets
        self.cifar10_train = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=train_transform)
        self.cifar10_test = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=test_transform)
        
        self.cifar100_train = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=train_transform)
        self.cifar100_test = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=test_transform)
        
        # DataLoaders
        self.cifar10_train_loader = DataLoader(
            self.cifar10_train, batch_size=self.batch_size, shuffle=True, num_workers=4)
        self.cifar10_test_loader = DataLoader(
            self.cifar10_test, batch_size=self.batch_size, shuffle=False, num_workers=4)
        
        self.cifar100_train_loader = DataLoader(
            self.cifar100_train, batch_size=self.batch_size, shuffle=True, num_workers=4)
        self.cifar100_test_loader = DataLoader(
            self.cifar100_test, batch_size=self.batch_size, shuffle=False, num_workers=4)
        
        print("Data setup completed!")
    
    def setup_models(self):
        """Initialisation des modèles CNN et ViT"""
        print("Setting up models...")
        
        # Modèles CNN
        self.models['resnet50'] = timm.create_model('resnet50', pretrained=False, num_classes=100)
        self.models['efficientnet_b0'] = timm.create_model('efficientnet_b0', pretrained=False, num_classes=100)
        
        # Modèles Vision Transformer
        self.models['vit_base_patch16_224'] = timm.create_model(
            'vit_base_patch16_224', pretrained=False, num_classes=100, img_size=32)
        self.models['deit_base_patch16_224'] = timm.create_model(
            'deit_base_patch16_224', pretrained=False, num_classes=100, img_size=32)
        
        # Déplacement vers GPU
        for name, model in self.models.items():
            self.models[name] = model.to(self.device)
            print(f"Model {name} loaded with {sum(p.numel() for p in model.parameters())} parameters")
    
    def train_model(self, model, train_loader, test_loader, model_type, model_name):
        """Entraînement d'un modèle spécifique"""
        print(f"Training {model_name} ({model_type})...")
        
        config = self.config[model_type]
        
        # Optimizer
        if config['optimizer'] == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=config['lr'], 
                                weight_decay=config['weight_decay'], momentum=0.9)
        else:  # AdamW
            optimizer = optim.AdamW(model.parameters(), lr=config['lr'], 
                                  weight_decay=config['weight_decay'])
        
        # Scheduler
        if config['scheduler'] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs)
        
        criterion = nn.CrossEntropyLoss()
        
        train_losses = []
        train_accuracies = []
        test_accuracies = []
        
        for epoch in range(self.num_epochs):
            # Warmup pour ViT
            if model_type == 'vit' and epoch < config['warmup_epochs']:
                lr_scale = min(1.0, float(epoch + 1) / config['warmup_epochs'])
                for param_group in optimizer.param_groups:
                    param_group['lr'] = config['lr'] * lr_scale
            
            # Phase d'entraînement
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader)):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            train_loss = running_loss / len(train_loader)
            train_acc = 100. * correct / total
            
            # Phase de test
            test_acc = self.evaluate_model(model, test_loader)
            
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)
            
            if scheduler:
                scheduler.step()
            
            print(f'Epoch: {epoch+1}/{self.num_epochs} | '
                  f'Loss: {train_loss:.4f} | '
                  f'Train Acc: {train_acc:.2f}% | '
                  f'Test Acc: {test_acc:.2f}%')
        
        return {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'test_accuracies': test_accuracies,
            'final_accuracy': test_accuracies[-1]
        }
    
    def evaluate_model(self, model, test_loader):
        """Évaluation du modèle sur le dataset de test"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        return 100. * correct / total
    
    def calculate_flops(self, model, input_size=(1, 3, 32, 32)):
        """Calcul des FLOPs (approximatif)"""
        from fvcore.nn import FlopCountAnalysis
        
        model.eval()
        input_tensor = torch.randn(input_size).to(self.device)
        
        flops = FlopCountAnalysis(model, input_tensor)
        return flops.total()
    
    def run_comparative_study(self):
        """Exécution de l'étude comparative complète"""
        print("Starting comparative study...")
        
        self.setup_data()
        self.setup_models()
        
        # Métriques de performance
        performance_metrics = {}
        
        for model_name, model in self.models.items():
            print(f"\n{'='*50}")
            print(f"Evaluating {model_name}")
            print(f"{'='*50}")
            
            # Détermination du type (CNN ou ViT)
            model_type = 'vit' if 'vit' in model_name or 'deit' in model_name else 'cnn'
            
            # Entraînement sur CIFAR-100
            start_time = time.time()
            results = self.train_model(model, self.cifar100_train_loader, 
                                     self.cifar100_test_loader, model_type, model_name)
            training_time = time.time() - start_time
            
            # Évaluation sur CIFAR-10
            cifar10_accuracy = self.evaluate_model(model, self.cifar10_test_loader)
            
            # Calcul des FLOPs et paramètres
            flops = self.calculate_flops(model)
            num_params = sum(p.numel() for p in model.parameters())
            
            # Mesure du temps d'inférence
            inference_time = self.measure_inference_time(model)
            
            # Stockage des résultats
            performance_metrics[model_name] = {
                'cifar10_accuracy': cifar10_accuracy,
                'cifar100_accuracy': results['final_accuracy'],
                'parameters': num_params,
                'flops': flops,
                'training_time': training_time,
                'inference_time': inference_time,
                'train_losses': results['train_losses'],
                'train_accuracies': results['train_accuracies'],
                'test_accuracies': results['test_accuracies']
            }
            
            print(f"\n{model_name} Results:")
            print(f"CIFAR-10 Accuracy: {cifar10_accuracy:.2f}%")
            print(f"CIFAR-100 Accuracy: {results['final_accuracy']:.2f}%")
            print(f"Parameters: {num_params:,}")
            print(f"FLOPs: {flops:,}")
            print(f"Training Time: {training_time/3600:.2f}h")
            print(f"Inference Time: {inference_time:.2f}ms")
        
        self.results = performance_metrics
        return performance_metrics
    
    def measure_inference_time(self, model, num_iterations=100):
        """Mesure du temps d'inférence moyen"""
        model.eval()
        input_tensor = torch.randn(1, 3, 32, 32).to(self.device)
        
        # Warmup
        for _ in range(10):
            _ = model(input_tensor)
        
        # Mesure
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(input_tensor)
        
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        end_time = time.time()
        
        return ((end_time - start_time) / num_iterations) * 1000  # en ms
    
    def generate_plots(self):
        """Génération des graphiques de comparaison"""
        if not self.results:
            print("No results to plot. Run the study first.")
            return
        
        # Configuration des plots
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Courbes d'entraînement (Accuracy)
        ax1 = axes[0, 0]
        for model_name, metrics in self.results.items():
            ax1.plot(metrics['train_accuracies'], label=model_name, linewidth=2)
        ax1.set_title('Training Accuracy Curves', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Accuracy (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Courbes de perte
        ax2 = axes[0, 1]
        for model_name, metrics in self.results.items():
            ax2.plot(metrics['train_losses'], label=model_name, linewidth=2)
        ax2.set_title('Training Loss Curves', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Comparaison des performances
        ax3 = axes[1, 0]
        model_names = list(self.results.keys())
        cifar100_accs = [self.results[name]['cifar100_accuracy'] for name in model_names]
        
        bars = ax3.bar(model_names, cifar100_accs, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax3.set_title('CIFAR-100 Test Accuracy Comparison', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Accuracy (%)')
        ax3.tick_params(axis='x', rotation=45)
        
        # Ajout des valeurs sur les barres
        for bar, acc in zip(bars, cifar100_accs):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 4. Comparaison efficacité (Accuracy vs Paramètres)
        ax4 = axes[1, 1]
        params = [self.results[name]['parameters'] / 1e6 for name in model_names]  # en millions
        accuracies = cifar100_accs
        
        scatter = ax4.scatter(params, accuracies, s=100, alpha=0.7)
        ax4.set_title('Accuracy vs Model Size', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Parameters (Millions)')
        ax4.set_ylabel('CIFAR-100 Accuracy (%)')
        ax4.grid(True, alpha=0.3)
        
        # Annotation des points
        for i, (param, acc, name) in enumerate(zip(params, accuracies, model_names)):
            ax4.annotate(name, (param, acc), xytext=(5, 5), textcoords='offset points')
        
        plt.tight_layout()
        plt.savefig('comparative_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_results_table(self):
        """Génération du tableau de résultats"""
        if not self.results:
            print("No results to display. Run the study first.")
            return
        
        data = []
        for model_name, metrics in self.results.items():
            data.append({
                'Model': model_name,
                'CIFAR-10 Acc (%)': f"{metrics['cifar10_accuracy']:.2f}",
                'CIFAR-100 Acc (%)': f"{metrics['cifar100_accuracy']:.2f}",
                'Params (M)': f"{metrics['parameters'] / 1e6:.1f}",
                'FLOPs (G)': f"{metrics['flops'] / 1e9:.2f}",
                'Training Time (h)': f"{metrics['training_time'] / 3600:.2f}",
                'Inference Time (ms)': f"{metrics['inference_time']:.2f}"
            })
        
        df = pd.DataFrame(data)
        print("\n" + "="*80)
        print("COMPARATIVE RESULTS SUMMARY")
        print("="*80)
        print(df.to_string(index=False))
        
        # Sauvegarde en CSV
        df.to_csv('comparative_results.csv', index=False)
        print(f"\nResults saved to 'comparative_results.csv'")
        
        return df

# Exécution de l'étude
if __name__ == "__main__":
    # Initialisation
    study = VisionComparativeStudy(batch_size=128, num_epochs=200)
    
    # Exécution de l'étude comparative
    results = study.run_comparative_study()
    
    # Génération des visualisations
    study.generate_plots()
    
    # Génération du tableau de résultats
    study.generate_results_table()
