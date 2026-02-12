from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import os
import torch
import numpy as np
from PIL import Image
import io
import base64
import torchvision.transforms as transforms
from .models import DefectLog

# Import our model
try:
    from src.models.model import SentinelModel
    from src.utils.gradcam import ExplainableAI
except ImportError:
    SentinelModel = None
    ExplainableAI = None

# Global model instance
model = None
explainable_ai = None
device = 'cpu'

def load_model():
    global model, explainable_ai, device
    if model is None and SentinelModel is not None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SentinelModel(model_name='resnet18', num_classes=2, pretrained=True)
        weights_path = os.path.join(settings.BASE_DIR, 'best_model.pth')
        if os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path, map_location=device))
        else:
            print("INFO: No custom weights found ('best_model.pth'). Using ImageNet pre-trained weights. Accuracy on specific defects may be low until fine-tuned.")
        model.to(device)
        model.eval()
        explainable_ai = ExplainableAI(model)

def index(request):
    # Calculate dynamic stats
    total_scanned = DefectLog.objects.count()
    defects_found = DefectLog.objects.filter(prediction='Defect').count()
    
    if total_scanned > 0:
        # Detection Rate = % of parts that are "No Defect" (or simply 1 - defect_rate)
        detection_rate = round(((total_scanned - defects_found) / total_scanned) * 100, 1)
    else:
        detection_rate = 0.0

    stats = {
        'total_scanned': total_scanned,
        'defects_found': defects_found,
        'detection_rate': detection_rate,
    }

    # Show recent logs in dashboard
    recent_logs = DefectLog.objects.all().order_by('-timestamp')[:5]
    return render(request, 'dashboard/index.html', {
        'recent_logs': recent_logs,
        'stats': stats
    })

def upload_image(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']
        
        # Save to DB immediately to get path, or handle manually. 
        # Easier to create instance first if we want simple file handling, 
        # but we need inference results to save the full object.
        # We'll use a temporary handler or save first then update.
        
        # Init log object
        log_entry = DefectLog(image=image_file, prediction='Pending', confidence=0.0)
        # We need to save to get the file on disk for inference usually, 
        # unless we pass the memory file to the transform.
        # Let's save it.
        log_entry.save() 
        
        file_path = log_entry.image.path
        uploaded_file_url = log_entry.image.url

        result = {
            'image_url': uploaded_file_url,
            'prediction': 'Unknown',
            'confidence': 0.0,
            'defect_prob': 0.0,
            'nodefect_prob': 0.0,
            'heatmap_b64': None,
            'error': None
        }

        # Run Inference
        load_model()
        if model:
            try:
                # Transform
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                pil_img = Image.open(file_path).convert('RGB')
                original_np = np.array(pil_img.resize((224, 224))).astype(np.float32) / 255.0
                
                input_tensor = transform(pil_img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    output = model(input_tensor)
                    probs = torch.softmax(output, dim=1)
                    # 0: Defect, 1: No Defect
                    defect_prob = probs[0][0].item() * 100
                    nodefect_prob = probs[0][1].item() * 100
                    print(f"DEBUG: Defect: {defect_prob}, No Defect: {nodefect_prob}")
                    
                    score, pred_idx = torch.max(probs, 1)
                
                labels = {0: "Defect", 1: "No Defect"}
                
                # Update Result
                pred_label = labels.get(pred_idx.item(), "Unknown")
                conf_val = round(score.item() * 100, 2)
                
                result['prediction'] = pred_label
                result['confidence'] = conf_val
                result['defect_prob'] = round(defect_prob, 2)
                result['nodefect_prob'] = round(nodefect_prob, 2)

                # Update DB
                log_entry.prediction = pred_label
                log_entry.confidence = conf_val
                log_entry.defect_prob = round(defect_prob, 2)
                log_entry.nodefect_prob = round(nodefect_prob, 2)
                log_entry.save()

                # Grad-CAM
                if explainable_ai:
                    _, heatmap = explainable_ai.generate_heatmap(input_tensor, original_np, target_class=pred_idx.item())
                    if heatmap is not None:
                        heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8))
                        buffered = io.BytesIO()
                        heatmap_img.save(buffered, format="PNG")
                        result['heatmap_b64'] = base64.b64encode(buffered.getvalue()).decode("utf-8")
                        
            except Exception as e:
                print(f"Inference error: {e}")
                result['error'] = str(e)
                # Cleanup if failed? Or keep as failed record.
                log_entry.prediction = "Error"
                log_entry.save()
        else:
             result['error'] = "Model not loaded"

        # Recalculate stats for the dashboard update
        total_scanned = DefectLog.objects.count()
        defects_found = DefectLog.objects.filter(prediction='Defect').count()
        detection_rate = round(((total_scanned - defects_found) / total_scanned) * 100, 1) if total_scanned > 0 else 0.0
        
        stats = {
            'total_scanned': total_scanned,
            'defects_found': defects_found,
            'detection_rate': detection_rate,
        }

        return render(request, 'dashboard/index.html', {
            'result': result, 
            'recent_logs': DefectLog.objects.all().order_by('-timestamp')[:5],
            'stats': stats
        })
    
    return index(request)

def reload_model_view(request):
    """Force reload of the model to pick up new weights."""
    global model, explainable_ai
    model = None
    explainable_ai = None
    load_model()
    
    # Also need stats here
    total_scanned = DefectLog.objects.count()
    defects_found = DefectLog.objects.filter(prediction='Defect').count()
    detection_rate = round(((total_scanned - defects_found) / total_scanned) * 100, 1) if total_scanned > 0 else 0.0
    stats = {'total_scanned': total_scanned, 'defects_found': defects_found, 'detection_rate': detection_rate}

    return render(request, 'dashboard/index.html', {
        'recent_logs': DefectLog.objects.all().order_by('-timestamp')[:5], 
        'message': 'Model reloaded successfully!',
        'stats': stats
    })
