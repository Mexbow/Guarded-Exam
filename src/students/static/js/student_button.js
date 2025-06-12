        function toggleAssignments(button) {
            const tableBody = button.closest('.card-body').querySelector('tbody');
            const hiddenRows = tableBody.querySelectorAll('tr.hidden');
            
            hiddenRows.forEach(row => row.classList.remove('hidden'));
            button.textContent = 'Show Less';
            button.onclick = () => {
                hiddenRows.forEach(row => row.classList.add('hidden'));
                button.textContent = 'Show More';
                button.onclick = () => toggleAssignments(button);
            };
        }