from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from .models import Exam, Question
from django.http import Http404
from .forms import QuestionForm 

def create_exam(request):
    if request.method == 'POST':
        # Get exam details
        title = request.POST.get('exam_title')
        duration = request.POST.get('exam_duration')  # Get duration from form
        code = request.POST.get('exam_code')  # If you have an exam code field
        
        # Validate duration
        try:
            duration = int(duration)
            if duration <= 0:
                messages.error(request, "Exam duration must be a positive number.")
                return render(request, 'exam_builder.html')
        except (ValueError, TypeError):
            messages.error(request, "Please enter a valid exam duration.")
            return render(request, 'exam_builder.html')
        
        # Create the exam
        exam = Exam.objects.create(
            title=title,
            duration=duration,  # Save duration in minutes
            code=code if code else None,  # Only set code if provided
            created_by=request.user
        )
        
        # Create questions
        question_count = int(request.POST.get('question_count', 0))
        
        if question_count == 0:
            messages.warning(request, "Exam created but no questions were added. Please add questions to make the exam functional.")
        
        for i in range(1, question_count + 1):
            question_text = request.POST.get(f'question_{i}_text')
            question_type = request.POST.get(f'question_{i}_type')
            options = request.POST.getlist(f'question_{i}_options')
            answer = request.POST.get(f'question_{i}_answer')

            # Skip empty questions
            if not question_text or not answer:
                continue

            # Handle question creation
            question = Question.objects.create(
                exam=exam,
                text=question_text,
                question_type=question_type,
                options=options if question_type == 'mcq' and options else None,
                answer=answer
            )
        
        messages.success(request, f"Exam '{title}' created successfully with {question_count} questions and {duration} minutes duration!")
        return redirect('view_exams')  # Adjust redirect to show the exams page

    return render(request, 'exam_builder.html')
def toggle_exam_status(request, exam_id):
    exam = get_object_or_404(Exam, id=exam_id, created_by=request.user)
    exam.is_active = not exam.is_active
    exam.save()
    messages.success(request, f"Exam {'activated' if exam.is_active else 'deactivated'} successfully!")
    return redirect('view_exams')

def view_exams(request):
    exams = Exam.objects.filter(created_by=request.user)
    return render(request, 'exams/view_exams.html', {'exams': exams})

# View the questions for a specific exam
def view_questions(request, exam_id):
    try:
        exam = Exam.objects.get(id=exam_id, created_by=request.user)
        questions = exam.questions.all()
        return render(request, 'exams/view_questions.html', {'exam': exam, 'questions': questions})
    except Exam.DoesNotExist:
        raise Http404("Exam not found")
        
def delete_exam(request, exam_id):
    exam = get_object_or_404(Exam, id=exam_id)

    # Check if the current user is the creator of the exam
    if exam.created_by == request.user:
        exam.delete()
        messages.success(request, "Exam deleted successfully!")
    else:
        messages.error(request, "You don't have permission to delete this exam.")
    
    return redirect('view_exams')

# edit the question
def edit_question(request, question_id):
    question = get_object_or_404(Question, id=question_id)

    if request.method == "POST":
        form = QuestionForm(request.POST, request.FILES, instance=question)
        if form.is_valid():
            form.save()
            return redirect('view_questions', exam_id=question.exam.id)  # Redirect back to the questions list for this exam
    else:
        form = QuestionForm(instance=question)

    return render(request, 'exams/edit_question.html', {'form': form, 'question': question})

def delete_question(request, question_id):
    question = get_object_or_404(Question, id=question_id)

    if request.method == "POST":
        exam_id = question.exam.id  # Save the exam ID before deleting
        question.delete()
        return redirect('view_questions', exam_id)

    return render(request, 'exams/confirm_delete_question.html', {'question': question})