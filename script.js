document.addEventListener('DOMContentLoaded', function() {
  let allPredictions = [];

  // Fetch model predictions from model_predictions.json
  fetch('model_predictions.json')
    .then(response => response.json())
    .then(data => {
      allPredictions = data;
      
      // Trigger initial update with the default date
      const dateInput = document.querySelector('#forecast-date');
      const defaultDate = dateInput.value;
      updateForecast(defaultDate, allPredictions);
      
      // Set up event listener for date changes
      dateInput.addEventListener('change', function() {
        updateForecast(this.value, allPredictions);
      });
    })
    .catch(error => {
      console.error('Error fetching model predictions:', error);
      const container = document.querySelector('.forecast-container');
      container.innerHTML = '<p>Error loading forecast data.</p>';
    });

  function updateForecast(selectedDateStr, allPredictions) {
    // Find the prediction object matching the selected date
    const prediction = allPredictions.find(p => p.date === selectedDateStr);
    
    const container = document.querySelector('.forecast-container');
    
    if (!prediction) {
      container.innerHTML = '<p>No forecast data available for the selected date.</p>';
      return;
    }
    
    // Clear existing forecast cards
    container.innerHTML = '';
    
    // Loop through 5 days (t+1 to t+5)
    for (let i = 1; i <= 5; i++) {
      // Calculate the future date
      const selectedDate = new Date(selectedDateStr);
      const futureDate = new Date(selectedDate);
      futureDate.setDate(futureDate.getDate() + i);
      
      // Get day of week
      const daysOfWeek = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
      const dayOfWeek = daysOfWeek[futureDate.getDay()];
      
      // Get temperature and icon from the correct keys
      const tempKey = `target_temp_t+${i}`;
      const iconKey = `icon_t+${i}`;
      const temperature = prediction[tempKey];
      const icon = prediction[iconKey];
      
      // Create forecast card HTML
      const cardHTML = `
        <div class="forecast-card">
          <h4>Day ${i}</h4>
          <img src="${icon}" alt="weather icon">
          <p class="temperature">${Math.round(temperature)}°C</p>
        </div>
      `;
      
      // Append card to container
      container.innerHTML += cardHTML;
    }
  }
});
