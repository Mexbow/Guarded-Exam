function updateTable(selectId, formId, actionPage) {
    var select = document.getElementById(selectId);
    var grade = select.value;
    var form = document.getElementById(formId);
    form.action = actionPage + '?grade=' + encodeURIComponent(grade);
    form.submit();
}

function updateTable_student() {
    updateTable('gradeSelect', 'gradeForm', 'student_control.php');
}

function updateTable_teacher() {
    updateTable('gradeSelect_teacher', 'gradeForm_teacher', 'class.php');
}

function updateTable_for_teacher() {
    updateTable('gradeSelect_for_teacher', 'gradeForm_for_teacher', 'teacher_to_grades.php');
}
function updateTable_for_teacher() {
    updateTable('gradeSelect_table', 'gradeForm_table', 'time_table.php');
}
function updateTable_for_teacher_assignment() {
    console.log('Grade selected:', document.getElementById('gradeSelect_teacher_assignment').value);
    updateTable('gradeSelect_teacher_assignment', 'gradeForm_teacher_assignment', 'admin_assignment.php');
}