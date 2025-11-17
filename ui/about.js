document.addEventListener('DOMContentLoaded', () => {
    let currentStep = 0;
    const totalSteps = 8;

    // Select all elements
    const steps = document.querySelectorAll('.progress-nav li');
    const contentSections = document.querySelectorAll('.content-section');
    const prevBtn = document.getElementById('prev-step');
    const nextBtn = document.getElementById('next-step');

    function updateView() {
        // Update progress bar text
        steps.forEach((step, index) => {
            step.classList.remove('active');
            if (index === currentStep) {
                step.classList.add('active');
            }
        });

        // Update content visibility
        contentSections.forEach((section, index) => {
            section.classList.remove('active');
            if (index === currentStep) {
                section.classList.add('active');
            }
        });

        // Update button states
        prevBtn.disabled = (currentStep === 0);
        nextBtn.disabled = (currentStep === totalSteps - 1);
    }

    // Add click listeners to buttons
    nextBtn.addEventListener('click', () => {
        if (currentStep < totalSteps - 1) {
            currentStep++;
            updateView();
        }
    });

    prevBtn.addEventListener('click', () => {
        if (currentStep > 0) {
            currentStep--;
            updateView();
        }
    });

    // Initialize the page
    updateView();
});