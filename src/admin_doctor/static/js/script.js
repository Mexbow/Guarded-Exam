const body = document.querySelector("body");
const darkLight = document.querySelector("#darkLight");
const sidebar = document.querySelector(".sidebar");
const submenuItems = document.querySelectorAll(".submenu_item");

// Check localStorage to see if dark mode is enabled
if (localStorage.getItem("theme") === "dark") {
  body.classList.add("dark");
  darkLight.classList.replace("bx-sun", "bx-moon");
} else {
  body.classList.remove("dark");
  darkLight.classList.replace("bx-moon", "bx-sun");
}

darkLight.addEventListener("click", () => {
  body.classList.toggle("dark");
  if (body.classList.contains("dark")) {
    localStorage.setItem("theme", "dark");
    darkLight.classList.replace("bx-sun", "bx-moon");
  } else {
    localStorage.setItem("theme", "light");
    darkLight.classList.replace("bx-moon", "bx-sun");
  }
});

submenuItems.forEach((item, index) => {
  item.addEventListener("click", () => {
    item.classList.toggle("show_submenu");
    submenuItems.forEach((item2, index2) => {
      if (index !== index2) {
        item2.classList.remove("show_submenu");
      }
    });
  });
});