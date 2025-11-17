document.addEventListener('DOMContentLoaded', () => {
    // Correctly parse the two required URL parameters
    const params = new URLSearchParams(window.location.search);
    const forecastDate = params.get('forecastDate');
    const horizon = params.get('horizon');

    if (!forecastDate || !horizon) {
        document.body.innerHTML = '<h1>Error: Missing required URL parameters. Please return to the main page.</h1>';
        return;
    }

    fetch('rich_data.json')
        .then(response => response.json())
        .then(allData => {
            const forecastData = allData.find(d => d.date === forecastDate);
            if (!forecastData) {
                document.body.innerHTML = `<h1>Error: No data found for the forecast date: ${forecastDate}</h1>`;
                return;
            }
            populateUI(forecastData, horizon);
        })
        .catch(error => {
            console.error('Error fetching data:', error);
            document.body.innerHTML = '<h1>Error loading weather data.</h1>';
        });
});

function populateUI(data, h) {
    // Dynamically create the keys for the selected horizon
    const tempKey = `target_temp_t+${h}`;
    const iconKey = `icon_t+${h}`;
    const humidityKey = `humidity_t+${h}`;
    const windKey = `windspeed_t+${h}`;
    const precipKey = `precip_t+${h}`;
    const uvKey = `uvindex_t+${h}`;
    const feelsLikeKey = `feelslike_t+${h}`;
    const precipProbKey = `precipprob_t+${h}`;

    // Calculate the actual date being displayed
    const baseDate = new Date(data.date + 'T12:00:00Z');
    baseDate.setDate(baseDate.getDate() + parseInt(h));
    
    // --- Left Panel ---
    document.getElementById('date-string').textContent = baseDate.toLocaleDateString('en-US', { weekday: 'long', month: 'short', day: 'numeric' });
    document.getElementById('main-temp').textContent = `${Math.round(data[tempKey])}°`;
    document.getElementById('main-weather-icon').src = data[iconKey];
    
    const weatherDescSpan = document.querySelector('#weather-description span');
    let description = "Weather";
    if (data[iconKey].includes('rainy')) description = 'Rainy';
    else if (data[iconKey].includes('cloudy')) description = 'Cloudy';
    else if (data[iconKey].includes('day') || data[iconKey].includes('sunny')) description = 'Sunny';
    else if (data[iconKey].includes('storm')) description = 'Storm';
    weatherDescSpan.textContent = description;

    // --- Dynamic Background ---
    const leftPanel = document.querySelector('.left-panel');
    let backgroundFile = 'day.png';
    if (data[iconKey].includes('storm')) backgroundFile = 'storm.png';
    else if (data[iconKey].includes('rainy')) backgroundFile = 'rainy.png';
    else if (data[iconKey].includes('cloudy')) backgroundFile = 'light_cloudy.png';
    else if (data[iconKey].includes('sunny') || data[iconKey].includes('day')) backgroundFile = 'sunny.png';
    leftPanel.style.backgroundImage = `url('images/backgrounds/${backgroundFile}')`;

    // --- Right Panel (Text Values) ---
    document.querySelector('#humidity-card .card-value').textContent = `${Math.round(data[humidityKey])}%`;
    document.querySelector('#wind-card .card-value').textContent = `${data[windKey]} km/h`;
    document.querySelector('#precip-card .card-value').textContent = `${(data[precipKey] / 10).toFixed(1)} cm`;
    document.querySelector('#uv-index-card .card-value').textContent = Math.round(data[uvKey]);
    document.querySelector('#feels-like-card .card-value').textContent = `${Math.round(data[feelsLikeKey])}°`;
    document.querySelector('#precip-prob-card .card-value').textContent = `${Math.round(data[precipProbKey])}%`;

    // ===============================================
    // --- NEW ROBUST DYNAMIC VISUALIZATION LOGIC ---
    // ===============================================

    // Humidity Card Logic
    const humidity = data[humidityKey];
    const humiditySegments = document.querySelectorAll('#humidity-card .bar-segment');
    humiditySegments.forEach(seg => seg.classList.remove('active')); // Reset all
    if (humidity < 60) {
        if (humiditySegments[0]) humiditySegments[0].classList.add('active'); // Good
    } else if (humidity < 80) {
        if (humiditySegments[1]) humiditySegments[1].classList.add('active'); // Normal
    } else {
        if (humiditySegments[2]) humiditySegments[2].classList.add('active'); // Bad
    }

    // UV Index Card Logic
    const uvIndex = Math.round(data[uvKey]);
    const uvSegments = document.querySelectorAll('#uv-index-card .bar-segment');
    const uvDesc = document.querySelector('#uv-index-card .card-description');
    uvSegments.forEach(seg => seg.classList.remove('active')); // Reset all
    if (uvIndex <= 2) { if (uvSegments[0]) uvSegments[0].classList.add('active'); if(uvDesc) uvDesc.textContent = 'Low'; }
    else if (uvIndex <= 5) { if (uvSegments[1]) uvSegments[1].classList.add('active'); if(uvDesc) uvDesc.textContent = 'Moderate'; }
    else if (uvIndex <= 7) { if (uvSegments[2]) uvSegments[2].classList.add('active'); if(uvDesc) uvDesc.textContent = 'High'; }
    else if (uvIndex <= 10) { if (uvSegments[3]) uvSegments[3].classList.add('active'); if(uvDesc) uvDesc.textContent = 'Very High'; }
    else { if (uvSegments[4]) uvSegments[4].classList.add('active'); if(uvDesc) uvDesc.textContent = 'Extreme'; }

    // Feels Like Slider Logic
    const feelsLikeTemp = data[feelsLikeKey];
    const feelsLikeThumb = document.querySelector('#feels-like-card .slider-thumb');
    if (feelsLikeThumb) {
        const feelsLikePercent = Math.max(0, Math.min(100, (feelsLikeTemp / 50) * 100));
        feelsLikeThumb.style.left = `${feelsLikePercent}%`;
    }

    // Chance of Rain Bar Logic
    const precipProb = data[precipProbKey];
    const rainFill = document.querySelector('#precip-prob-card .slider-fill');
    if (rainFill) {
        rainFill.style.width = `${precipProb}%`;
    }

    // Precipitation Dot Scale Logic
    const precipValue = data[precipKey]; // in mm
    const precipDots = document.querySelectorAll('#precip-card .scale-dot');
    precipDots.forEach((dot, index) => {
        const dotValue = index * 10;
        if (dotValue <= precipValue) {
            dot.classList.add('active');
        } else {
            dot.classList.remove('active');
        }
    });

    // --- Back Button ---
    document.getElementById('back-button').href = 'index.html';
    
    // --- Create Hourly Chart ---
    createHourlyChart(data);
}

function createHourlyChart(data) {
    const labels = [];
    const temperatures = [];
    
    // Loop through 24 hours
    for (let h = 0; h < 24; h++) {
        // Add time label (e.g., "00:00", "01:00")
        labels.push(`${h.toString().padStart(2, '0')}:00`);
        
        // Get predicted temperature for this hour
        const tempKey = `pred_temp_h${h}`;
        temperatures.push(data[tempKey]);
    }
    
    // Get canvas element
    const ctx = document.getElementById('hourly-chart').getContext('2d');
    
    // Create Chart.js instance
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                data: temperatures,
                backgroundColor: 'rgba(74, 140, 247, 0.2)',
                borderColor: '#4A8CF7',
                fill: true,
                tension: 0.4,
                pointRadius: 0
            }]
        },
        options: {
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                x: {
                    display: false
                },
                y: {
                    display: false
                }
            }
        }
    });
}
