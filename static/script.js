const darkModeToggle = document.getElementById("darkModeToggle");
const loadingOverlay = document.getElementById("loadingOverlay");
const forms = document.querySelectorAll("form");

function applySavedTheme() {
    const savedTheme = localStorage.getItem("theme");
    if (savedTheme === "dark") {
        document.body.classList.add("dark-mode");
    }
}

darkModeToggle.addEventListener("click", () => {
    document.body.classList.toggle("dark-mode");
    const theme = document.body.classList.contains("dark-mode") ? "dark" : "light";
    localStorage.setItem("theme", theme);
});

forms.forEach((form) => {
    form.addEventListener("submit", () => {
        loadingOverlay.classList.add("show");
    });
});

applySavedTheme();
