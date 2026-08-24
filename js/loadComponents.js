// Utility to fetch and inject HTML components
function loadComponent(id, file, callback) {
  fetch(file)
    .then(res => res.text())
    .then(data => {
      const el = document.getElementById(id);
      if (!el) return;

      el.innerHTML = data;
      if (callback) callback();
    })
    .catch(err => console.error(`Failed to load ${file}:`, err));
}

// Sets the active state on navbar links based on current route
function setActiveNav() {
  let currentPath = window.location.pathname;

  if (currentPath === "/") currentPath = "/index.html";

  const links = document.querySelectorAll(".navbar-links a");

  // Map nested routes (e.g. blog posts) to parent nav item
  const routeMap = {
    "/blogs/": "/blogs.html",
  };

  links.forEach(link => {
    const linkPath = new URL(link.href).pathname;
    let isActive = false;

    for (const prefix in routeMap) {
      if (
        currentPath.startsWith(prefix) &&
        linkPath === routeMap[prefix]
      ) {
        isActive = true;
        break;
      }
    }

    if (currentPath === linkPath) {
      isActive = true;
    }

    link.classList.toggle("active", isActive);
  });
}

// Initializes shared layout (navbar + footer)
function initLayout() {
  loadComponent("navbar", "/components/navbar.html", () => {
    setActiveNav();

    // Initialize navbar-dependent UI after injection
    if (window.UI) {
      UI.initNavbarToggle();
      UI.initThemeToggle();
    }
  });

  loadComponent("footer", "/components/footer.html", () => {
    if (window.UI) {
      UI.initLastUpdated();
    }
  });
}

// Entry point
document.addEventListener("DOMContentLoaded", initLayout);