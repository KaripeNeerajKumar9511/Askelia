from django.db import models


class ReportAnalysis(models.Model):
    patient_name = models.CharField(max_length=200)
    age = models.IntegerField()
    gender = models.CharField(max_length=50)
    bp = models.CharField(max_length=50, blank=True, null=True)
    diabetic = models.CharField(max_length=50, blank=True, null=True)
    hyperthyroidism = models.CharField(max_length=50, blank=True, null=True)
    health_issues = models.TextField(blank=True, null=True)

    # ✅ Use TextField for MySQL if you’re unsure about JSON support
    report_urls = models.TextField()   # we’ll store JSON string
    ai_result = models.TextField()     # full AI result as JSON string

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.patient_name} ({self.age}, {self.gender})"
