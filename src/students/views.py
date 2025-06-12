from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.utils.timezone import now
from django.contrib.auth.models import User
from random import shuffle
from .models import Student, StudentExamAttempt, StudentAnswer, AIAnalysisLog
from exams.models import Exam, Question
from .model import *
import torch
import time
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def enter_exam_code(request):
    if request.method == "POST":
        student_name = request.POST.get("student_name", "").strip()
        student_id = request.POST.get("student_id", "").strip()
        exam_code = request.POST.get("exam_code", "").strip()

        if not student_name or not student_id:
            messages.error(request, "Please enter your name and ID.")
            return redirect("enter_exam_code")

        try:
            exam = Exam.objects.get(code=exam_code)
            if not exam.is_active:
                messages.error(request, "This exam is not available anymore or is currently inactive.")
                return render(request, 'students/enter_exam_code.html')
            request.session["student_name"] = student_name
            request.session["student_id"] = student_id
            return redirect("take_exam", exam_id=exam.id)
        except Exam.DoesNotExist:
            messages.error(request, "Invalid exam code.")

    return render(request, "students/enter_exam_code.html")

def take_exam(request, exam_id):
    exam = get_object_or_404(Exam, id=exam_id)
    questions = list(Question.objects.filter(exam=exam))
    shuffle(questions)

    student_name = request.session.get("student_name", "").strip()
    student_id = request.session.get("student_id", "").strip()

    if not student_name or not student_id:
        messages.error(request, "Invalid student session. Please enter again.")
        return redirect("enter_exam_code")

    # Optimize user and student creation
    user, _ = User.objects.get_or_create(username=student_id, defaults={"first_name": student_name})
    student, _ = Student.objects.get_or_create(user=user, defaults={"student_id": student_id, "full_name": student_name})

    if student.full_name != student_name:
        student.full_name = student_name
        student.save()

    if StudentExamAttempt.objects.filter(student=student, exam=exam).exists():
        messages.warning(request, "You have already attempted this exam.")
        return redirect("enter_exam_code")

    if request.method == "POST":
        attempt = StudentExamAttempt.objects.create(student=student, exam=exam, score=0, start_time=now())
        score = 0
        total_ai_detected = 0
        total_flagged_answers = 0

        for question in questions:
            answer_key = f"answer_{question.id}"
            student_answer = request.POST.get(answer_key, "").strip()

            if question.question_type == "text":
                print(f"DEBUG: Evaluating text answer - Student: {student_answer}, Correct: {question.answer}")
                
                # Initialize variables with default values
                ai_prediction = 0
                ai_confidence = 0.0
                similarity_score = 0.0
                is_ai_generated = False
                is_correct = False
                result_message = ""
                processing_time = 0.0
                
                analysis_start_time = time.time()
                
                try:
                    # Get AI detection results - USE THE SAME FUNCTION AS YOUR MANUAL TESTING
                    ai_prediction, ai_confidence = predict_logistic(
                        models, ai_detection_meta_model, student_answer, tokenizers, device=device
                    )
                    
                    # Fix the AI detection logic - make it consistent with your manual testing
                    # Based on your manual test: pred is True for AI, False for Human
                    is_ai_generated = bool(ai_prediction)  # True means AI-generated, False means human
                    
                    print(f"DEBUG: AI Detection - prediction: {ai_prediction}, confidence: {ai_confidence}, is_ai_generated: {is_ai_generated}")
                    print(f"DEBUG: Prediction interpretation - {'AI' if ai_prediction else 'Human'}")
                    
                    # Get similarity score
                    similarity_score = compute_similarity_logistic(
                        student_answer, question.answer, similarity_meta_model
                    )
                    
                    print(f"DEBUG: Similarity Score: {similarity_score:.2f}%")
                    
                    # Determine if answer is correct based on similarity threshold AND not AI-generated
                    similarity_threshold = 60  # You can make this configurable
                    is_correct = (similarity_score >= similarity_threshold) and (not is_ai_generated)
                    
                    # Create the result message - using ASCII alternatives
                    if is_ai_generated:
                        result_message = "[AI DETECTED] Answer detected as AI-generated."
                        total_ai_detected += 1
                        total_flagged_answers += 1
                        print(f"DEBUG: AI DETECTED - Answer flagged as AI-generated")
                    elif similarity_score >= similarity_threshold:
                        result_message = f"[CORRECT] Answer is correct. Similarity: {similarity_score:.2f}%"
                        print(f"DEBUG: CORRECT - High similarity, human-written")
                    else:
                        result_message = f"[INCORRECT] Answer is incorrect. Similarity: {similarity_score:.2f}%"
                        print(f"DEBUG: INCORRECT - Low similarity")
                    
                    analysis_end_time = time.time()
                    processing_time = analysis_end_time - analysis_start_time
                    
                    print(f"DEBUG: Final result: {result_message}")
                    
                except Exception as e:
                    print(f"ERROR in AI analysis: {e}")
                    # Fallback values
                    ai_prediction = 0
                    ai_confidence = 0.0
                    similarity_score = 0.0
                    is_ai_generated = False
                    is_correct = False
                    result_message = "[ERROR] Error in analysis: " + str(e)
                    processing_time = 0.0
                
            else:
                # Handle MCQ questions
                is_correct = student_answer.lower() == (question.answer or "").lower()
                is_ai_generated = False
                ai_prediction = 0
                ai_confidence = None
                similarity_score = None
                result_message = "MCQ Answer"
                processing_time = 0.0

            if is_correct:
                score += 1

            # Create StudentAnswer with detailed AI analysis data
            student_answer_obj = StudentAnswer.objects.create(
                attempt=attempt,
                question=question,
                student_answer=student_answer,
                is_correct=is_correct,
                # Store the detailed AI analysis results
                ai_detection_result=result_message,
                ai_confidence_score=ai_confidence,
                similarity_score=similarity_score,
                is_ai_generated=is_ai_generated,  # This should now be correctly set
                original_score=is_correct
            )
            
            print(f"DEBUG: Stored answer with is_ai_generated: {is_ai_generated}")
            
            # Create detailed AI analysis log for text questions
            if question.question_type == "text":
                try:
                    # Prepare AI detection data with proper type conversion
                    ai_detection_data = {
                        'prediction': bool(ai_prediction),  # Store as boolean for consistency
                        'confidence': float(ai_confidence) if ai_confidence is not None else None,
                        'is_ai_generated': bool(is_ai_generated),
                        'prediction_text': 'AI' if ai_prediction else 'Human'  # Add text interpretation
                    }
                    
                    # Prepare similarity data with proper type conversion
                    similarity_data = {
                        'similarity_score': float(similarity_score) if similarity_score is not None else None,
                        'threshold': 60,  # Use the actual threshold value
                        'is_similar': bool(similarity_score >= 60 if similarity_score is not None else False)
                    }
                    
                    AIAnalysisLog.objects.create(
                        student_answer=student_answer_obj,
                        ai_detection_raw=json.dumps(ai_detection_data, ensure_ascii=True),
                        similarity_raw=json.dumps(similarity_data, ensure_ascii=True),
                        processing_time=float(processing_time),
                        model_version="RoBERTa+GPT2+DeBERTa_Ensemble_v1.0",
                        error_log=None
                    )
                    print(f"DEBUG: Successfully created AI analysis log for question {question.id}")
                except Exception as log_error:
                    print(f"ERROR creating AI analysis log: {log_error}")
                    # Create a simplified log entry if the detailed one fails
                    try:
                        AIAnalysisLog.objects.create(
                            student_answer=student_answer_obj,
                            ai_detection_raw=f"AI Generated: {is_ai_generated}, Confidence: {ai_confidence}",
                            similarity_raw=f"Similarity: {similarity_score}%",
                            processing_time=float(processing_time),
                            model_version="RoBERTa+GPT2+DeBERTa_Ensemble_v1.0",
                            error_log=str(log_error)
                        )
                    except Exception as fallback_error:
                        print(f"ERROR creating fallback AI analysis log: {fallback_error}")

        # Update attempt with AI analysis summary
        attempt.score = score
        attempt.ai_analysis_completed = True
        attempt.total_ai_detected = total_ai_detected
        attempt.total_flagged_answers = total_flagged_answers
        attempt.save()
        
        print(f"DEBUG: Final summary - Total AI detected: {total_ai_detected}, Total flagged: {total_flagged_answers}")
        
        return redirect("exam_congrats", attempt_id=attempt.id)

    return render(request, "students/take_exam.html", {"exam": exam, "questions": questions, "duration": exam.duration})

def exam_results(request):
    attempts = StudentExamAttempt.objects.all().order_by("-submitted_at")
    return render(request, "students/exam_results.html", {"attempts": attempts})

def exam_congrats(request, attempt_id):
    attempt = get_object_or_404(StudentExamAttempt, id=attempt_id)
    answers = StudentAnswer.objects.filter(attempt=attempt)
    return render(request, "students/exam_congrats.html", {"attempt": attempt, "answers": answers})