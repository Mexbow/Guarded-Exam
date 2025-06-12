document.addEventListener("DOMContentLoaded", function () {
  var iframeContainer = document.getElementById("iframeContainer");

  function handleButtonClick(event) {
    var iframe = document.createElement("iframe");
    var targetId = event.currentTarget.id;
    switch (targetId) {
      case "openButton_student":
        iframe.src = "add_student.php";
        break;
      case "openButton_class":
        iframe.src = "teacher_to_class.php";
        break;
      case "openButton":
        iframe.src = "add_teacher.php";
        break;
      case "openButton_table":
        iframe.src = "add_table.php";
        break;
      case "openButton_assignment":
        iframe.src = "add_assignment.php";
        break;
      case "openButton_zoom":
        iframe.src = "add_zoom.php";
        break;
      case "openButton_upgrade_student":
        iframe.src = "upgrade.php"; 
        break;
      case "openButton_teacher_edit":
        iframe.src = "edit_teacher.php";  
        break;
      default:
        iframe.src = "";
        break;
    }

    iframeContainer.innerHTML = ""; 
    iframeContainer.appendChild(iframe);
    iframeContainer.style.display = "block";
  }

  var openButtonStudent = document.getElementById("openButton_student");
  var openButtonClass = document.getElementById("openButton_class");
  var openButton = document.getElementById("openButton");
  var openButtonTable = document.getElementById("openButton_table");
  var openButtonAssignment = document.getElementById("openButton_assignment");
  var openButtonZoom = document.getElementById("openButton_zoom");
  var openButtonUpgradeStudent = document.getElementById("openButton_upgrade_student"); 
  var openButtonTeacherEdit = document.getElementById("openButton_teacher_edit");

  if (openButtonStudent) {
    openButtonStudent.addEventListener("click", handleButtonClick);
  }
  if (openButtonClass) {
    openButtonClass.addEventListener("click", handleButtonClick);
  }
  if (openButton) {
    openButton.addEventListener("click", handleButtonClick);
  }
  if (openButtonTable) {
    openButtonTable.addEventListener("click", handleButtonClick);
  }
  if (openButtonAssignment) {
    openButtonAssignment.addEventListener("click", handleButtonClick);
  }
  if (openButtonZoom) {
    openButtonZoom.addEventListener("click", handleButtonClick);
  }
  if (openButtonUpgradeStudent) {
    openButtonUpgradeStudent.addEventListener("click", handleButtonClick);
  }
  if (openButtonTeacherEdit) {
    openButtonTeacherEdit.addEventListener("click", handleButtonClick);
  }

  window.closeIframe = function () {
    iframeContainer.style.display = "none";
  };
});
