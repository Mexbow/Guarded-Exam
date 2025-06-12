$(document).ready(function() {
    $('#gradeSelect').change(function() {
        var selectedGrade = $(this).val();
        if (selectedGrade) {
            $.ajax({
                url: 'fetch_classes.php', // PHP file to fetch classes
                type: 'POST',
                data: { grade: selectedGrade },
                success: function(data) {
                    $('#classSelect').html(data);
                },
                error: function() {
                    console.error('Failed to fetch classes');
                }
            });
        } else {
            $('#classSelect').html('<option value="" disabled selected>Select Class</option>');
        }
    });
});
