
let animatedTitle = document.querySelector("#animatedTitle");
const text = "OXFORD";
const speed = 300;
const delay = 850;
let i = 0;

function typeWriter() {
    if (i < text.length) {
        if (animatedTitle.children[i]) {
            animatedTitle.children[i].style.visibility = "visible";
        }
        i++;
        setTimeout(typeWriter, speed);
    } else {
        setTimeout(() => {
            Array.from(animatedTitle.children).forEach(span => {
                span.style.visibility = "hidden";
            });
            i = 0;
            setTimeout(typeWriter, delay);
        }, delay);
    }
}

if (animatedTitle && animatedTitle.children.length > 0) {
    typeWriter();
} else {
    console.error("animatedTitle element not found or has no children");
}




//  -------------------------------------------------
const inputField = document.getElementById('search');
const std_search = document.getElementById('search_std');
const resultsTableBody = document.querySelector('#phpBody');
const stdBody = document.querySelector('#std_body');
// Search teachers
if(inputField){
inputField.addEventListener('keyup', async () => {
    const value = inputField.value;

    const selectedTable = 'teachers'; 
    const selectedCol = 'teacher_name'; // Column to search in the teachers table

    try {
        const response = await fetch('search.php', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: new URLSearchParams({ 
                query: value,
                table: selectedTable,
                col: selectedCol
            }).toString(),
        });

        if (response.ok) {
            const result = await response.json();

            resultsTableBody.innerHTML = ''; // Clear previous results

            result.forEach(item => {
                const row = document.createElement('tr');

                row.innerHTML = `
                    <th scope="row">${item.teacher_id}</th>
                    <td>${item.teacher_name}</td>
                    <td>${item.teacher_email}</td>
                    <td>${item.teacher_pass}</td>
                    <td>${item.teacher_address}</td>
                    <td>${item.teacher_phone}</td>
                    <td>
                        <button class="btn btn-primary">
                            <a href="update_teacher.php?updateid=${item.user_id}" class="text-light">UPDATE</a>
                        </button>
                        <button class="btn btn-danger">
                            <a href="delete_user.php?deleteid=${item.user_id}&source=control" class="text-light">DELETE</a>
                        </button>
                    </td>
                `;

                resultsTableBody.appendChild(row);
            });

        } else {
            console.error('Error sending data:', response.statusText);
        }
    } catch (error) {
        console.error('Fetch error:', error);
    }
});
}

// Search teacher_class

const inputTeacherClass = document.getElementById('search_teacher_class');
const teacherClassBody = document.getElementById('teacherClassBody');

if (inputTeacherClass) {
    inputTeacherClass.addEventListener('keyup', async () => {
        const value = inputTeacherClass.value.trim();

        if (value === '') {
            teacherClassBody.innerHTML = ''; 
            return;
        }

        const selectedTable = 'teacher_class';
        const selectedCol = 'teacher';

        try {
            const response = await fetch('search.php', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: new URLSearchParams({
                    query: value,
                    table: selectedTable,
                    col: selectedCol
                }).toString(),
            });

            if (response.ok) {
                const result = await response.json();
                console.log('Fetch Result:', result); // Log the result to see what is returned

                teacherClassBody.innerHTML = ''; // Clear previous results

                if (result.length > 0) {
                    result.forEach(item => {
                        const row = document.createElement('tr');

                        row.innerHTML = `
                            <th scope="row">${item.teacher}</th>
                            <td>${item.grade}</td>
                            <td>${item.class}</td>
                            <td>${item.teacher_dep}</td>
                            <td>${item.teacher_subject}</td>
                            <td>
                                <button class="btn btn-primary">
                                    <a href="update_class.php?updateid=${item.class_id}" class="text-light">UPDATE</a>
                                </button>
                                 <button class="btn btn-danger">
                                <a href="delete_class.php?delete=${item.class_id}" class="text-light">Delete</a>
                            </button>
                            </td>
                        `;

                        teacherClassBody.appendChild(row);
                    });
                } else {
                    teacherClassBody.innerHTML = '<tr><td colspan="5">No results found</td></tr>';
                }

            } else {
                console.error('Error sending data:', response.statusText);
            }
        } catch (error) {
            console.error('Fetch error:', error);
        }
    });
}

// Search students
if(std_search){
std_search.addEventListener('keyup', async () => {
    const value = std_search.value;
    const selectedTable = 'students'; 
    const selectedCol = 'student_name'; 

    try {
        const response = await fetch('search.php', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: new URLSearchParams({ 
                query: value,
                table: selectedTable,
                col: selectedCol
            }).toString(),
        });

        if (response.ok) {
            const result = await response.json();

            stdBody.innerHTML = ''; // Clear previous results

            result.forEach(item => {
                const row = document.createElement('tr');

                row.innerHTML = `
                    <th scope="row" class="text-nowrap">${item.student_id}</th>
                    <td class="text-nowrap">${item.student_name}</td>
                    <td class="text-nowrap">${item.student_email}</td>
                    <td class="text-nowrap">${item.student_pass}</td>
                    <td class="text-nowrap">${item.Grade}</td>
                    <td class="text-nowrap">${item.Class}</td>
                    <td class="text-nowrap">${item.student_dep}</td>
                    <td class="text-nowrap">${item.religion}</td>                   
                    <td class="text-nowrap">${item.student_phone}</td>
                    <td class="text-nowrap">${item.bus_info}</td>
                    <td>
                        <div class="btn-group" role="group">
                            <button class="btn btn-primary">
                                <a href="update_student.php?updateid=${item.user_id}" class="text-light">UPDATE</a>
                            </button>
                            <button class="btn btn-danger">
                                <a href="delete_user.php?deleteid=${item.user_id}&source=student" class="text-light">DELETE</a>
                            </button>
                        </div>
                    </td>
                `;

                stdBody.appendChild(row);
            });

        } else {
            console.error('Error sending data:', response.statusText);
        }
    } catch (error) {
        console.error('Fetch error:', error);
    }
});
}
// Search Student_Fee
const input_fee = document.getElementById('search_fee');
const std_fee = document.getElementById('std_fee');

if(input_fee){
    input_fee.addEventListener('keyup', async () => {
        const value = input_fee.value;
        const selectedTable = 'students'; 
        const selectedCol = 'student_name'; 
    
        try {
            const response = await fetch('search.php', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: new URLSearchParams({ 
                    query: value,
                    table: selectedTable,
                    col: selectedCol
                }).toString(),
            });
    
            if (response.ok) {
                const result = await response.json();
    
                std_fee.innerHTML = ''; // Clear previous results
    
                result.forEach(item => {
                    // Encode values for use in URL parameters
                    const encodedUserId = encodeURIComponent(item.user_id);
                    const leftToPay = (parseFloat(item.educational_fee) + parseFloat(item.book_fee) + parseFloat(item.bus_fee)) - parseFloat(item.paid);
                    const encodedLeftToPay = encodeURIComponent(leftToPay);

                    const row = document.createElement('tr');
    
                    row.innerHTML = `
                        <th scope="row" class="text-nowrap">${item.student_id}</th>
                        <td class="text-nowrap">${item.student_name}</td>
                        <td class="text-nowrap">${item.Grade}</td>
                        <td class="text-nowrap">${item.Class}</td>
                        <td class="text-nowrap">${item.educational_fee}</td>
                        <td class="text-nowrap">${item.book_fee}</td>
                        <td class="text-nowrap">${item.bus_fee}</td>                   
                        <td class="text-nowrap">${item.paid}</td>
                        <td>
                            <div class="btn-group" role="group">
                                <button class="btn btn-primary">
                                    <a href="update_fee.php?updateid=${encodedUserId}" class="text-light">Update</a>
                                </button>
                                <button class="btn btn-warning">
                                    <a href="send_notif.php?left_pay=${encodedLeftToPay}&user_id=${encodedUserId}" class="text-light">Send</a>
                                </button>
                            </div>
                        </td>
                    `;
    
                    std_fee.appendChild(row); // Appending to std_fee
                });
    
            } else {
                console.error('Error sending data:', response.statusText);
            }
        } catch (error) {
            console.error('Fetch error:', error);
        }
    });
}
// Search Assignments
const inputAssignmentSearch = document.getElementById('searchAssignments');
const assignmentTableBody = document.getElementById('assignmentsTableBody');

if (inputAssignmentSearch) {
    inputAssignmentSearch.addEventListener('keyup', async () => {
        const value = inputAssignmentSearch.value.trim();

        if (value === '') {
            assignmentTableBody.innerHTML = ''; 
            return;
        }

        try {
            const response = await fetch('search.php', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: new URLSearchParams({
                    query: value,
                    table: 'assignments',
                    col: 'teacher_name' // This will be handled in PHP
                }).toString(),
            });

            if (response.ok) {
                const result = await response.json();
                console.log('Fetch Result:', result); // Log the result to see what is returned

                assignmentTableBody.innerHTML = ''; // Clear previous results

                if (result.length > 0) {
                    result.forEach(item => {
                        const row = document.createElement('tr');

                        row.innerHTML = `
                            <td>${item.assignment_name}</td>
                            <td>${item.teacher_name}</td>
                            <td>${item.due_date}</td>
                            <td>${item.grade}</td>
                            <td>
                                <a href='admin_delete_assignment.php?id=${item.assignment_id}' class='btn btn-danger btn-sm'>Delete</a>
                            </td>
                        `;

                        assignmentTableBody.appendChild(row);
                    });
                } else {
                    assignmentTableBody.innerHTML = '<tr><td colspan="5">No results found</td></tr>';
                }

            } else {
                console.error('Error sending data:', response.statusText);
            }
        } catch (error) {
            console.error('Fetch error:', error);
        }
    });
}



    
    var notificationIcon = document.getElementById("notificationIcon");
    var notificationDropdown = document.getElementById("notificationDropdown");
    var notificationCount = document.getElementById("notificationCount");

    // Function to toggle the notification dropdown
    notificationIcon.addEventListener("click", function () {
      if (notificationDropdown.style.display === "none" || notificationDropdown.style.display === "") {
        notificationDropdown.style.display = "block";
      } else {
        notificationDropdown.style.display = "none";
      }
    });

    // Hide dropdown when clicking outside
    document.addEventListener("click", function (event) {
      if (!notificationDropdown.contains(event.target) && event.target !== notificationIcon) {
        notificationDropdown.style.display = "none";
      }
    });



    async function fetchUnreadCount() {
        try {
            const response = await fetch('selectNotif.php', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: new URLSearchParams({
                    action: 'count_unread',
                }).toString(),
            });
    
            if (response.ok) {
                const result = await response.json();
                const unreadCount = result.unread_count;
                if (notificationCount) {
                    notificationCount.textContent = unreadCount;
                }
            } else {
                console.error('Error fetching unread count:', response.statusText);
            }
        } catch (error) {
            console.error('Fetch error:', error);
        }
    }
    
    // Call fetchUnreadCount on page load or at desired intervals
    fetchUnreadCount();
    if (notificationIcon) {
        notificationIcon.addEventListener("click", async () => {
            try {
                // Mark all notifications as read
                
    
                // Fetch notifications after marking them as read
                const response = await fetch('selectNotif.php', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/x-www-form-urlencoded',
                    },
                    body: new URLSearchParams({
                        action: 'fetch_notifications', 
                    }).toString(),
                });
    
                if (response.ok) {
                    const result = await response.json();
                    const notifications = result.notifications;
    
                    notificationDropdown.innerHTML = ''; 
    
                    if (notifications.length > 0) {
                        const notificationList = document.createElement('ul'); // Create a single <ul> to hold all notifications
    
                        notifications.forEach(item => {
                            const listItem = document.createElement('li');
    
                            listItem.style.backgroundColor = item.is_read === 0 ? '#ffffe0' : '#ffffff';
                            listItem.style.fontWeight = item.is_read === 0 ? 'bold' : 'normal';
    
                            listItem.innerHTML = `
                                ${item.content}
                            `;
    
                            notificationList.prepend(listItem); // Prepend each <li> to the <ul> to ensure the newest are at the top
                        });
    
                        notificationDropdown.appendChild(notificationList); // Append the <ul> to the dropdown
                    } else {
                        notificationDropdown.innerHTML = `
                            <li>No notifications found</li>
                        `;
                    }
    
                    const seeMoreLink = document.createElement('a');
                    seeMoreLink.href = "notification_page.php";
                    seeMoreLink.textContent = "See more";
                    seeMoreLink.style.setProperty('color', 'red');
                    seeMoreLink.style.textDecoration = 'none';
                    seeMoreLink.style.fontWeight = 'bold';
                    seeMoreLink.style.display = 'block';
                    seeMoreLink.style.textAlign = 'center';
                    notificationDropdown.appendChild(seeMoreLink);
    
                } else {
                    console.error('Error fetching data:', response.statusText);
                }
                await markAllAsRead();
            } catch (error) {
                console.error('Fetch error:', error);
            }
        });
    }
    
    // Function to mark all notifications as read
    async function markAllAsRead() {
        try {
            const response = await fetch('selectNotif.php', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: new URLSearchParams({
                    action: 'mark_all_as_read',
                }).toString(),
            });
    
            if (!response.ok) {
                console.error('Error marking all notifications as read:', response.statusText);
            }
        } catch (error) {
            console.error('Fetch error:', error);
        }
    }
    