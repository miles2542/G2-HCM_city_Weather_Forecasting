document.addEventListener('DOMContentLoaded', () => {
    // --- STEP 1: Get the required parameters from the URL ---
    const params = new URLSearchParams(window.location.search);
    const forecastDate = params.get('forecastDate');
    const horizon = params.get('horizon');

    // If parameters are missing, we can't do anything.
    if (!forecastDate || !horizon) {
        document.body.innerHTML = '<h1>Error: Missing required URL parameters. Please return to the main page.</h1>';
        return;
    }

    // --- STEP 2: Fetch the DAILY data for the main display and cards ---
    fetch('rich_data.json')
        .then(response => {
            if (!response.ok) throw new Error('Could not load rich_data.json');
            return response.json();
        })
        .then(allDailyData => {
            // Find the specific forecast run we need from the daily data file
            const dailyData = allDailyData.find(d => d.date === forecastDate);
            if (!dailyData) {
                document.body.innerHTML = `<h1>Error: No daily data found for forecast date: ${forecastDate}</h1>`;
                return;
            }
            
            // Use this data to populate everything EXCEPT the chart
            populateUiCardsAndPanel(dailyData, horizon);

            // --- STEP 3: Fetch the HOURLY data for the chart ---
            // First, calculate the specific date we need to show details for
            const detailDate = new Date(forecastDate + 'T12:00:00Z');
            detailDate.setDate(detailDate.getDate() + parseInt(horizon));
            const detailDateStr = detailDate.toISOString().split('T')[0];

            fetch('hourly_predictions.json')
                .then(response => {
                    if (!response.ok) throw new Error('Could not load hourly_predictions.json');
                    return response.json();
                })
                .then(allHourlyData => {
                    // Find the specific day's data in the hourly file
                    const hourlyData = allHourlyData.find(d => d.date === detailDateStr);
                    if (hourlyData) {
                        createHourlyChart(hourlyData);
                    } else {
                        console.error(`No hourly data found for date: ${detailDateStr}`);
                        // Optionally display a message in the hourly chart area
                        const hourlyContainer = document.querySelector('.hourly-forecast-card');
                        if (hourlyContainer) hourlyContainer.innerHTML = '<p>No hourly data available for this day.</p>';
                    }
                })
                .catch(error => console.error('Error fetching hourly data:', error));
        })
        .catch(error => {
            console.error('Error loading daily data:', error);
            document.body.innerHTML = '<h1>Error loading weather data.</h1>';
        });
});


/**
 * This is YOUR working function. It populates everything except the hourly chart.
 * I have renamed it for clarity.
 */
function populateUiCardsAndPanel(data, h) {
    const tempKey = `target_temp_t+${h}`;
    const iconKey = `icon_t+${h}`;
    const humidityKey = `humidity_t+${h}`;
    const windKey = `windspeed_t+${h}`;
    const precipKey = `precip_t+${h}`;
    const uvKey = `uvindex_t+${h}`;
    const feelsLikeKey = `feelslike_t+${h}`;
    const precipProbKey = `precipprob_t+${h}`;

    const baseDate = new Date(data.date + 'T12:00:00Z');
    baseDate.setDate(baseDate.getDate() + parseInt(h));
    
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

    const leftPanel = document.querySelector('.left-panel');
    let backgroundFile = 'day.png';
    if (data[iconKey].includes('storm')) backgroundFile = 'storm.png';
    else if (data[iconKey].includes('rainy')) backgroundFile = 'rainy.png';
    else if (data[iconKey].includes('cloudy')) backgroundFile = 'light_cloudy.png';
    else if (data[iconKey].includes('sunny') || data[iconKey].includes('day')) backgroundFile = 'sunny.png';
    leftPanel.style.backgroundImage = `url('images/backgrounds/${backgroundFile}')`;

    document.querySelector('#humidity-card .card-value').textContent = `${Math.round(data[humidityKey])}%`;
    document.querySelector('#wind-card .card-value').textContent = `${data[windKey]} km/h`;
    document.querySelector('#precip-card .card-value').textContent = `${(data[precipKey] / 10).toFixed(1)} cm`;
    document.querySelector('#uv-index-card .card-value').textContent = Math.round(data[uvKey]);
    document.querySelector('#feels-like-card .card-value').textContent = `${Math.round(data[feelsLikeKey])}°`;
    document.querySelector('#precip-prob-card .card-value').textContent = `${Math.round(data[precipProbKey])}%`;

    const humidity = data[humidityKey];
    const humiditySegments = document.querySelectorAll('#humidity-card .bar-segment');
    humiditySegments.forEach(seg => seg.classList.remove('active'));
    if (humidity < 60) {
        if (humiditySegments[0]) humiditySegments[0].classList.add('active');
    } else if (humidity < 80) {
        if (humiditySegments[1]) humiditySegments[1].classList.add('active');
    } else {
        if (humiditySegments[2]) humiditySegments[2].classList.add('active');
    }

    const uvIndex = Math.round(data[uvKey]);
    const uvSegments = document.querySelectorAll('#uv-index-card .bar-segment');
    const uvDesc = document.querySelector('#uv-index-card .card-description');
    uvSegments.forEach(seg => seg.classList.remove('active'));
    if (uvIndex <= 2) { if (uvSegments[0]) uvSegments[0].classList.add('active'); if(uvDesc) uvDesc.textContent = 'Low'; }
    else if (uvIndex <= 5) { if (uvSegments[1]) uvSegments[1].classList.add('active'); if(uvDesc) uvDesc.textContent = 'Moderate'; }
    else if (uvIndex <= 7) { if (uvSegments[2]) uvSegments[2].classList.add('active'); if(uvDesc) uvDesc.textContent = 'High'; }
    else if (uvIndex <= 10) { if (uvSegments[3]) uvSegments[3].classList.add('active'); if(uvDesc) uvDesc.textContent = 'Very High'; }
    else { if (uvSegments[4]) uvSegments[4].classList.add('active'); if(uvDesc) uvDesc.textContent = 'Extreme'; }

    const feelsLikeTemp = data[feelsLikeKey];
    const feelsLikeThumb = document.querySelector('#feels-like-card .slider-thumb');
    if (feelsLikeThumb) {
        const feelsLikePercent = Math.max(0, Math.min(100, (feelsLikeTemp / 50) * 100));
        feelsLikeThumb.style.left = `${feelsLikePercent}%`;
    }

    const precipProb = data[precipProbKey];
    const rainFill = document.querySelector('#precip-prob-card .slider-fill');
    if (rainFill) {
        rainFill.style.width = `${precipProb}%`;
    }

    const precipValue = data[precipKey];
    const precipDots = document.querySelectorAll('#precip-card .scale-dot');
    precipDots.forEach((dot, index) => {
        const dotValue = index * 10;
        if (dotValue <= precipValue) {
            dot.classList.add('active');
        } else {
            dot.classList.remove('active');
        }
    });

    document.getElementById('back-button').href = 'index.html';
}


/**
 * This new function creates the rich hourly chart display
 */
function createHourlyChart(hourlyData) {
    const timelineContainer = document.getElementById('hourly-details-timeline');
    const canvas = document.getElementById('hourly-chart');

    if (!timelineContainer || !canvas) {
        console.error('Hourly chart containers not found.');
        return;
    }

    // Prepare arrays for both the timeline and the chart data
    const labels = [];
    const temperatures = [];
    let timelineHTML = ''; // We will build the HTML for the timeline here

    // Loop through all 24 hours of the day
    for (let h = 0; h < 24; h++) {
        const timeLabel = `${String(h).padStart(2, '0')}:00`;
        const temp = Math.round(hourlyData[`pred_temp_h${h}`]);
        const icon = hourlyData[`icon_h${h}`];

        // Add data for the chart's line graph
        labels.push(timeLabel);
        temperatures.push(temp);

        // Build the HTML for one item in the timeline above the chart
        timelineHTML += `
            <div class="hourly-item">
                <p class="time">${timeLabel}</p>
                <img src="${icon}" alt="weather icon" class="icon">
                <p class="temp">${temp}°</p>
                <div class="vertical-line"></div>
            </div>
        `;
    }

    // Inject the complete timeline HTML into the container
    timelineContainer.innerHTML = timelineHTML;

    // --- Create the Chart.js line graph ---
    const ctx = canvas.getContext('2d');
    
    // Destroy any previous chart instance to prevent errors
    if (window.myHourlyChart instanceof Chart) {
        window.myHourlyChart.destroy();
    }

    // Create the new chart
    window.myHourlyChart = new Chart(ctx, {
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
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: { enabled: false }
            },
            scales: {
                x: { display: false },
                y: {
                    display: false,
                    min: Math.min(...temperatures) - 3,
                    // THIS IS THE FIX:
                    suggestedMax: Math.max(...temperatures) + 3 
                }
            }
        }
    });
}
