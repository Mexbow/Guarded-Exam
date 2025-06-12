from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.models import User
from .models import Doctor
from django.contrib.auth.decorators import user_passes_test, login_required
from django.contrib.auth import authenticate, login
from django.contrib import messages
from students.models import StudentExamAttempt, Student
from students.models import StudentExamAttempt, StudentAnswer, AIAnalysisLog
from exams.models import Exam, ExamSubmission
from students.models import Student, StudentExamAttempt
from django.contrib.auth.views import PasswordResetView, PasswordResetDoneView, PasswordResetConfirmView, PasswordResetCompleteView
from django.urls import reverse_lazy
from django.contrib.auth.tokens import default_token_generator
from django.contrib.sites.shortcuts import get_current_site
from django.core.mail import send_mail
from django.template.loader import render_to_string
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from django.utils.encoding import force_bytes, force_str
from django.http import HttpResponse
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError

# Home View
def home_view(request):
    return render(request, 'admin_doctor/home.html')

# Check if the user is an admin
def is_admin(user):
    return user.is_superuser

# Admin Dashboard View
@login_required
@user_passes_test(is_admin)
def admin_dashboard(request):
    context = {
        'total_doctors' :Doctor.objects.count(),
        'total_students': Student.objects.count(),
        'total_exams': Exam.objects.count(),
    }  
    # Pass the count to the template
    return render(request, 'admin_doctor/admin_dashboard.html',context)

# Show all doctors
@login_required
@user_passes_test(is_admin)
def show_all_doctors(request):
    doctors = Doctor.objects.all()
    return render(request, 'admin_doctor/show_doctors.html', {'doctors': doctors})

# Check if the user is a doctor
def is_doctor(user):
    return hasattr(user, 'doctor')

# View Doctor Profile
def view_doctor(request, doctor_id):
    doctor = get_object_or_404(Doctor, id=doctor_id)
    return render(request, 'admin_doctor/view_doctor.html', {'doctor': doctor})

# Edit Doctor Profile
def edit_doctor(request, doctor_id):
    doctor = get_object_or_404(Doctor, id=doctor_id)

    if request.method == 'POST':
        new_email = request.POST.get('email')
        
        # Check if the new email is already taken by another user
        if User.objects.filter(email=new_email).exclude(id=doctor.user.id).exists():
            messages.error(request, f"A user with email {new_email} already exists.")
            return render(request, 'admin_doctor/edit_doctor.html', {'doctor': doctor})
        
        # Update doctor details
        doctor.user.first_name = request.POST.get('first_name')
        doctor.user.last_name = request.POST.get('last_name')
        
        # Update both email and username since they should match
        doctor.user.email = new_email
        doctor.user.username = new_email  # Keep username and email in sync
        
        doctor.phone = request.POST.get('phone')
        doctor.user.save()
        doctor.save()
        
        messages.success(request, f"Doctor {doctor.user.first_name} {doctor.user.last_name} has been updated successfully.")
        return redirect('show_all_doctors')  # Redirect to doctor list

    return render(request, 'admin_doctor/edit_doctor.html', {'doctor': doctor})

# Delete Doctor
@login_required
@user_passes_test(is_admin)
def delete_doctor(request, doctor_id):
    doctor = get_object_or_404(Doctor, id=doctor_id)
    doctor.user.delete()  # Deletes the user and cascades to the doctor profile
    messages.success(request, f"Doctor {doctor.user.first_name} {doctor.user.last_name} has been deleted.")
    return redirect('show_all_doctors')

# Add a New Doctor
@user_passes_test(is_admin)
def add_doctor(request):
    if request.method == 'POST':
        first_name = request.POST.get('first_name')
        last_name = request.POST.get('last_name')
        email = request.POST.get('email')
        phone = request.POST.get('phone')

        # Check if user with this email already exists
        if User.objects.filter(email=email).exists():
            messages.error(request, f"A user with email {email} already exists.")
            return render(request, 'admin_doctor/add_doctor.html')
        
        # Also check if username exists (should be the same as email, but just in case)
        if User.objects.filter(username=email).exists():
            messages.error(request, f"A user with username {email} already exists.")
            return render(request, 'admin_doctor/add_doctor.html')

        try:
            # Create the user with email as both username and email
            user = User.objects.create_user(
                username=email,  # Using email as username
                email=email,     # Setting email field
                first_name=first_name,
                last_name=last_name,
                password='doctor'  # Default password, should be changed later
            )

            # Create the doctor profile
            Doctor.objects.create(user=user, phone=phone)

            messages.success(request, f"Doctor {first_name} {last_name} added successfully! Username: {email}")
            return redirect('show_all_doctors')
        except Exception as e:
            messages.error(request, f"An error occurred while adding the doctor: {e}")
            return render(request, 'admin_doctor/add_doctor.html')

    return render(request, 'admin_doctor/add_doctor.html')

@login_required
@user_passes_test(is_doctor)
def doctor_view_results(request):
    attempts = StudentExamAttempt.objects.all().order_by("-submitted_at")

    return render(request, "students/doctor_results.html", {
        "attempts": attempts
    })

# Doctor Dashboard View
@login_required
@user_passes_test(is_doctor)
def doctor_dashboard(request):
    # Add any relevant logic for the doctor dashboard here
    return render(request, 'admin_doctor/doctor_dashboard.html')

# Success Page View
@login_required
def success(request):
    return render(request, 'admin_doctor/success.html')

# Login View
def user_login(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')
        user = authenticate(request, username=email, password=password)

        if user is not None:
            login(request, user)
            if user.is_superuser:
                return redirect('admin_dashboard')
            elif hasattr(user, 'doctor'):
                return redirect('doctor_dashboard')
        else:
            return render(request, 'admin_doctor/login.html', {'error': 'Invalid credentials'})
    return render(request, 'admin_doctor/login.html')

def student_exam_log(request, attempt_id):
    attempt = get_object_or_404(StudentExamAttempt, id=attempt_id)
    answers = StudentAnswer.objects.filter(attempt=attempt).select_related('question')
    
    # Calculate statistics
    total_questions = attempt.exam.questions.count()
    correct_answers = answers.filter(is_correct=True).count()
    ai_detected_answers = answers.filter(is_ai_generated=True).count()
    percentage = round((correct_answers / total_questions) * 100) if total_questions > 0 else 0
    
    context = {
        'attempt': attempt,
        'answers': answers,
        'total_questions': total_questions,
        'correct_answers': correct_answers,
        'ai_detected_answers': ai_detected_answers,
        'percentage': percentage,
    }
    
    return render(request, 'admin_doctor/student_exam_log.html', context)

@login_required
def update_answer_score(request, answer_id):
    if request.method == 'POST':
        answer = get_object_or_404(StudentAnswer, id=answer_id)
        is_correct = request.POST.get('is_correct') == 'true'
        doctor_notes = request.POST.get('doctor_notes', '')
        
        # Update the answer
        answer.is_correct = is_correct
        answer.doctor_notes = doctor_notes
        answer.last_modified_by = request.user
        answer.manual_override = True
        answer.save()
        
        # Recalculate the attempt score
        attempt = answer.attempt
        attempt.score = StudentAnswer.objects.filter(attempt=attempt, is_correct=True).count()
        attempt.save()
        
        messages.success(request, 'Answer score updated successfully.')
        return redirect('student_exam_log', attempt_id=attempt.id)
    
    messages.error(request, 'Invalid request.')
    return redirect('doctor_results')

# PASSWORD RESET VIEWS FOR DOCTORS

def doctor_password_reset_request(request):
    """Custom password reset request view for doctors only"""
    if request.method == 'POST':
        email = request.POST.get('email')
        try:
            user = User.objects.get(email=email)
            # Check if user is a doctor
            if hasattr(user, 'doctor'):
                # Generate password reset token
                token = default_token_generator.make_token(user)
                uid = urlsafe_base64_encode(force_bytes(user.pk))
                
                # Get current site
                current_site = get_current_site(request)
                
                # Prepare email context
                context = {
                    'user': user,
                    'domain': current_site.domain,
                    'uid': uid,
                    'token': token,
                    'protocol': 'https' if request.is_secure() else 'http',
                }
                
                # Render email template
                subject = 'Password Reset Request - Doctor Portal'
                email_template_name = 'admin_doctor/password_reset_email.html'
                email_content = render_to_string(email_template_name, context)
                
                # Send email
                send_mail(
                    subject,
                    email_content,
                    'guardedexam@gmail.com', 
                    [user.email],
                    html_message=email_content,
                    fail_silently=False,
                )
                
                messages.success(request, 'Password reset email has been sent to your email address.')
                return redirect('password_reset_done')
            else:
                messages.error(request, 'No doctor account found with this email address.')
        except User.DoesNotExist:
            messages.error(request, 'No account found with this email address.')
    
    return render(request, 'admin_doctor/password_reset_form.html')

def doctor_password_reset_done(request):
    """Password reset email sent confirmation"""
    return render(request, 'admin_doctor/password_reset_done.html')

def doctor_password_reset_confirm(request, uidb64, token):
    """Password reset confirmation view"""
    try:
        uid = force_str(urlsafe_base64_decode(uidb64))
        user = User.objects.get(pk=uid)
    except (TypeError, ValueError, OverflowError, User.DoesNotExist):
        user = None
    
    if user is not None and default_token_generator.check_token(user, token):
        if request.method == 'POST':
            password1 = request.POST.get('new_password1')
            password2 = request.POST.get('new_password2')
            
            if password1 and password1 == password2:
                try:
                    # Use Django's built-in password validation
                    validate_password(password1, user)
                    
                    # If validation passes, set the new password
                    user.set_password(password1)
                    user.save()
                    messages.success(request, 'Your password has been reset successfully.')
                    return redirect('password_reset_complete')
                    
                except ValidationError as errors:
                    # Display all validation errors
                    for error in errors:
                        messages.error(request, error)
                        
            elif not password1:
                messages.error(request, 'Password cannot be empty.')
            else:
                messages.error(request, 'Passwords do not match.')
        
        return render(request, 'admin_doctor/password_reset_confirm.html', {
            'validlink': True,
            'user': user
        })
    else:
        return render(request, 'admin_doctor/password_reset_confirm.html', {
            'validlink': False
        })

def doctor_password_reset_complete(request):
    """Password reset completion view"""
    return render(request, 'admin_doctor/password_reset_complete.html')