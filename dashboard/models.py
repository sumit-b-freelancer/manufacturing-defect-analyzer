from django.db import models

class DefectLog(models.Model):
    image = models.ImageField(upload_to='inspections/')
    prediction = models.CharField(max_length=50)
    confidence = models.FloatField()
    defect_prob = models.FloatField(default=0.0)
    nodefect_prob = models.FloatField(default=0.0)
    timestamp = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-timestamp']

    def __str__(self):
        return f"{self.prediction} ({self.confidence}%) - {self.timestamp}"
