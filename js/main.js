// Encapsulates all UI-related logic
window.UI = (function () {

    // Handles mobile navbar toggle
	function initNavbarToggle() {
		const toggle = document.getElementById("navbar-toggle");
		const links  = document.getElementById("navbar-links");

		if (!toggle || !links) return;

		function openMenu() {
		links.classList.add("open");
		toggle.setAttribute("aria-expanded", "true");
		}
		function closeMenu() {
		links.classList.remove("open");
		toggle.setAttribute("aria-expanded", "false");
		}
		function isOpen() {
		return links.classList.contains("open");
		}

		// Hamburger click
		toggle.addEventListener("click", (e) => {
		e.stopPropagation();
		isOpen() ? closeMenu() : openMenu();
		});

		// Close when a nav link is clicked
		links.addEventListener("click", (e) => {
		if (e.target.tagName === "A") closeMenu();
		});

		// Close when clicking outside the navbar
		document.addEventListener("click", (e) => {
		if (isOpen() && !e.target.closest(".navbar")) closeMenu();
		});

		// Close on Escape key
		document.addEventListener("keydown", (e) => {
		if (e.key === "Escape" && isOpen()) closeMenu();
		});
	}

	// Handles theme switching and persistence
	function initThemeToggle() {
		const toggleBtn = document.getElementById("theme-toggle");
		const body = document.body;

		if (!toggleBtn) return;

		// Apply saved theme on load
		const savedTheme = localStorage.getItem("theme");
		if (savedTheme === "light") {
		body.classList.add("light-theme");
		} else {
		body.classList.remove("light-theme");
		}

		// Toggle on click
		toggleBtn.addEventListener("click", () => {
		body.classList.toggle("light-theme");
		const isLight = body.classList.contains("light-theme");
		localStorage.setItem("theme", isLight ? "light" : "dark");
		});
	}

	// Image Modal for profile picture 
	function initImageModal() {
		const modal = document.getElementById("image-modal");
		const modalImg = document.getElementById("enlarged-image");
		const closeBtn = document.getElementById("close-modal");

		if (!modal || !modalImg || !closeBtn) return;

		document.querySelectorAll(".modal-image").forEach(img => {
			img.addEventListener("click", () => {
				modal.style.display = "block";
				modalImg.src = img.src;
				modalImg.alt = img.alt;
			});
		});

		closeBtn.addEventListener("click", () => {
			modal.style.display = "none";
		});

		modal.addEventListener("click", (e) => {
			if (e.target === modal) {
				modal.style.display = "none";
			}
		});
	}


	// // Profile Image Slideshow 
	// function initProfileSlideshow() {
	//   const img = document.getElementById("profile-image");
	//   const title = document.querySelector(".image-title");

	//   if (!img) return;

	//   const slides = [
	//     {
	//       src: "/images/udaipur.PNG",
	//       title: "Rajasthan, India ' 2024"
	//     },
	//     {
	//       src: "/images/arpon-kapuria-face-gemini.png",
	//       title: "Gemini Enhanced Portrait"
	//     }
	//   ];

	//   let index = 0;
	//   let interval = null;

	//   // Preload images
	//   slides.forEach((slide) => {
	//     const preload = new Image();
	//     preload.src = slide.src;
	//   });

	//   function showSlide(i) {
	//     img.style.opacity = 0;

	//     setTimeout(() => {
	//       img.src = slides[i].src;

	//       if (title) {
	//         title.textContent = slides[i].title;
	//       }

	//       img.style.opacity = 1;
	//     }, 300);
	//   }

	//   function start() {
	//     stop();

	//     interval = setInterval(() => {
	//       index = (index + 1) % slides.length;
	//       showSlide(index);
	//     }, 5000);
	//   }

	//   function stop() {
	//     clearInterval(interval);
	//   }

	//   // Initial slide
	//   showSlide(index);

	//   // Start slideshow
	//   start();

	//   // Pause on hover
	//   img.addEventListener("mouseenter", stop);
	//   img.addEventListener("mouseleave", start);
	// }

	// Renders gallery and initializes related features
	function initGallery() {
		const galleryGrid = document.getElementById("gallery-grid");
		if (!galleryGrid) return;

		const images = [
		{ src: 'travel/India/IMG_2337.jpg', country: 'in', title: 'Church Street', caption: 'June 2024 • Bangalore, India' },
			{ src: 'travel/India/IMG_3933.jpg', country: 'in', title: 'South Bombay', caption: 'June 2023 • Mumbai, India' },
			{ src: 'travel/India/IMG_4057.jpg', country: 'in', title: 'Marine Drive', caption: 'June 2023 • Mumbai, India' },
			{ src: 'travel/India/IMG_4460.jpg', country: 'in', title: 'IIT Bombay', caption: 'July 2023 • Mumbai, India' },
			{ src: 'travel/India/IMG_9496.jpg', country: 'in', title: 'Patrika Gate', caption: 'December 2023 • Jaipur, India' },
			{ src: 'travel/India/IMG_9590.jpg', country: 'in', title: 'Amber Fort', caption: 'December 2023 • Jaipur, India' },
			{ src: 'travel/Malaysia/IMG_5017.jpg', country: 'my', title: 'Petronas Twin Tower', caption: 'July 2024 • Kuala Lumpur, Malaysia' },
			{ src: 'travel/Malaysia/IMG_5029.jpg', country: 'my', title: 'Pavilion, Bukit Bintang', caption: 'July 2024 • Kuala Lumpur, Malaysia' },
			{ src: 'travel/Malaysia/IMG_5567.jpg', country: 'my', title: 'Cenang Beach', caption: 'July 2024 • Langkawi, Malaysia' },
		];

		galleryGrid.innerHTML = images.map(img => `
		<div class="gallery-item" data-country="${img.country}">
			<img src="${img.src}" alt="${img.title}" />
			<div class="gallery-overlay">
			<div class="gallery-caption">
				<h3>${img.title}</h3>
				<p>${img.caption}</p>
			</div>
			</div>
		</div>
		`).join("");

		initGalleryModal();
		initGalleryFilters();
	}

	// Gallery image modal
	function initGalleryModal() {
		const modal = document.getElementById("gallery-image-modal");
		const modalImg = document.getElementById("gallery-enlarged-image");
		const closeBtn = document.getElementById("gallery-close-modal");

		if (!modal || !modalImg || !closeBtn) return;

		document.querySelectorAll(".gallery-item img").forEach(img => {
		img.addEventListener("click", () => {
			modal.style.display = "block";
			modalImg.src = img.src;
		});
		});

		closeBtn.onclick = () => modal.style.display = "none";

		modal.onclick = (e) => {
		if (e.target === modal) modal.style.display = "none";
		};
	}

	// Filters gallery items by country
	function initGalleryFilters() {
		const buttons = document.querySelectorAll(".filter-btn");
		const items = document.querySelectorAll(".gallery-item");

		if (!buttons.length) return;

		function getCountryFromURL() {
		const params = new URLSearchParams(window.location.search);
		return params.get("country") || "all";
		}

		function applyFilter(country) {
		items.forEach(item => {
			const match =
			country === "all" ||
			item.getAttribute("data-country") === country;

			item.style.display = match ? "" : "none";
		});
		}

		function updateURL(country) {
		const params = new URLSearchParams();
		if (country !== "all") params.set("country", country);

		window.history.replaceState({}, "", `${window.location.pathname}?${params}`);
		}

		const initial = getCountryFromURL();
		applyFilter(initial);

		buttons.forEach(btn => {
		if (btn.dataset.country === initial) {
			btn.classList.add("active");
		}

		btn.addEventListener("click", () => {
			buttons.forEach(b => b.classList.remove("active"));
			btn.classList.add("active");

			const country = btn.dataset.country;
			applyFilter(country);
			updateURL(country);
		});
		});
	}

	// Filters articles by selected tags
	function initArticleFilters() {
		const buttons = document.querySelectorAll(".filter-button");
		const articles = document.querySelectorAll(".article-item");

		if (!buttons.length) return;

		const activeFilters = new Set();

		buttons.forEach(button => {
		button.addEventListener("click", () => {
			const filter = button.dataset.filter;

			if (activeFilters.has(filter)) {
			activeFilters.delete(filter);
			button.classList.remove("active");
			} else {
			activeFilters.add(filter);
			button.classList.add("active");
			}

			articles.forEach(article => {
			const tags = article.dataset.tags?.split(" ") || [];

			const visible =
				activeFilters.size === 0 ||
				[...activeFilters].some(f => tags.includes(f));

			article.style.display = visible ? "" : "none";
			});
		});
		});
	}

	// Fetches and renders the "last updated" date in the footer, with 1-day caching
	function initLastUpdated() {
		const el = document.getElementById("last-updated");
		if (!el) return;

		const cached = localStorage.getItem("lastUpdated");
		const cachedAt = localStorage.getItem("lastUpdatedAt");
		const oneDay = 24 * 60 * 60 * 1000;

		if (cached && cachedAt && Date.now() - cachedAt < oneDay) {
			el.textContent = cached;
			return;
		}

		fetch("https://api.github.com/repos/arponkapuria/arponkapuria.github.io/commits?per_page=1")
			.then(res => res.json())
			.then(data => {
				const date = new Date(data[0].commit.committer.date);
				const formatted = date.toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" });
				el.textContent = formatted;
				localStorage.setItem("lastUpdated", formatted);
				localStorage.setItem("lastUpdatedAt", Date.now());
			})
			.catch(() => {
				el.textContent = "recently";
			});
	}

	// Calculates blog reading time dynamically
	function initReadingTime() {

		// Main article container
		const article =
		document.querySelector(".blog-post");

		// Element where reading time will appear
		const readingTimeElement =
		document.getElementById("reading-time");

		if (!article || !readingTimeElement) return;

		// Get all readable text
		const text = article.innerText;

		// Count words
		const words =
		text.trim().split(/\s+/).length;

		// Count code blocks
		const codeBlocks =
		article.querySelectorAll("pre code").length;

		// Count images
		const images =
		article.querySelectorAll("img").length;

		// Base reading speed
		let readingTime =
		Math.ceil(words / 220);

		// Technical adjustments
		readingTime += codeBlocks * 1;
		readingTime += Math.ceil(images * 0.3);

		// Minimum 1 minute
		readingTime = Math.max(1, readingTime);

		// Add 1 minute buffer
		readingTime += 1;

		// Render
		readingTimeElement.innerText =
		`${readingTime} min read`;
	}

	// Initializes page-specific features
	function initPage() {
		initNavbarToggle();
		initThemeToggle();
		initImageModal();
		// initProfileSlideshow();
		initGallery();
		initArticleFilters();
		initReadingTime();
	}

	return {
		initNavbarToggle,
		initThemeToggle,
		initLastUpdated,
		initPage
	};

})();

// Entry point for page-level features
document.addEventListener("DOMContentLoaded", () => {
  UI.initPage();
});