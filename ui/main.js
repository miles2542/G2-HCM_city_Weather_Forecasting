// main.js — load data, update UI, handle interactions

let allPredictions = []; // Global store for all fetched prediction data

document.addEventListener('DOMContentLoaded', () => {
    // --- Get references to all DOM elements ---
    const datePicker = document.getElementById('forecast-date');
    const todayTempEl = document.getElementById('today-temp');
    const todayLinkEl = document.getElementById('today-weather-link');
    const forecastContainer = document.querySelector('.forecast-container');
    const appBackground = document.getElementById('app-background');
    const prevDayBtn = document.getElementById('prev-day-btn');
    const nextDayBtn = document.getElementById('next-day-btn');

    // --- Helper function to add days to a date string ---
    function addDays(dateStr, days) {
        const d = new Date(dateStr + 'T12:00:00Z'); // Use a neutral timezone to avoid off-by-one errors
        d.setDate(d.getDate() + days);
        return d.toISOString().split('T')[0]; // Format back to YYYY-MM-DD
    }

    // --- Helper function to format a date for display ---
    function formatDateLabel(isoDate) {
        const d = new Date(isoDate + 'T12:00:00Z');
        return d.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' });
    }

    // --- BUG FIX 1: Correctly updates the background ---
    function updateBackground(iconPath) {
        let backgroundFile = 'day.png'; // Safe default
        if (!iconPath) {
            appBackground.style.backgroundImage = `url('images/backgrounds/${backgroundFile}')`;
            return;
        }
        if (iconPath.includes('storm')) backgroundFile = 'storm.png';
        else if (iconPath.includes('rainy')) backgroundFile = 'rainy.png';
        else if (iconPath.includes('cloudy')) backgroundFile = 'light_cloudy.png';
        else if (iconPath.includes('sunny') || iconPath.includes('day')) backgroundFile = 'sunny.png';
        
        appBackground.style.backgroundImage = `url('images/backgrounds/${backgroundFile}')`;
    }

    // --- Main function to update the entire UI ---
    function updateDisplay(selectedDateStr) {
        // This is now much simpler: just find the data for the selected date.
        const dataObj = allPredictions.find(item => item.date === selectedDateStr);

        if (!dataObj) {
            todayTempEl.textContent = 'N/A';
            todayLinkEl.href = '#';
            forecastContainer.innerHTML = '<p style="color: white; opacity: 0.8;">No forecast data available for this date.</p>';
            updateBackground(null);
            return;
        }

        // MAIN DISPLAY ("Today" - uses t+0 data)
        todayTempEl.textContent = `${Math.round(dataObj['target_temp_t+0'])}°C`;
        todayLinkEl.href = `detail.html?forecastDate=${dataObj.date}&horizon=0`;
        updateBackground(dataObj['icon_t+0']);

        // NEXT 5 DAYS (uses t+1 to t+5 data)
        forecastContainer.innerHTML = '';
        for (let h = 1; h <= 5; h++) {
            const cardLink = document.createElement('a');
            cardLink.className = 'forecast-card';
            cardLink.href = `detail.html?forecastDate=${dataObj.date}&horizon=${h}`;

            const futureDate = addDays(selectedDateStr, h);
            
            cardLink.innerHTML = `
                <h4>${formatDateLabel(futureDate)}</h4>
                <img src="${dataObj[`icon_t+${h}`]}" alt="">
                <div class="temperature">${Math.round(dataObj[`target_temp_t+${h}`])}°C</div>
            `;
            forecastContainer.appendChild(cardLink);
        }
    }
    
    // --- Fetch data and initialize the application ---
    fetch('./rich_data.json')
        .then(resp => {
            if (!resp.ok) throw new Error(`Failed to fetch rich_data.json: ${resp.status}`);
            return resp.json();
        })
        .then(data => {
            if (!Array.isArray(data)) throw new Error('Expected an array in rich_data.json');
            allPredictions = data;

            if (datePicker && datePicker.value) {
                updateDisplay(datePicker.value);
            } else if (allPredictions.length > 0) {
                const fallbackDate = allPredictions[0].date;
                if (datePicker) datePicker.value = fallbackDate;
                updateDisplay(fallbackDate);
            }
        })
        .catch(err => {
            console.error('Error loading predictions:', err);
            if (todayTempEl) todayTempEl.textContent = 'N/A';
            if (forecastContainer) forecastContainer.innerHTML = '<p style="color:rgba(255,255,255,0.8)">Unable to load forecast data.</p>';
        });

    // --- Event Listeners ---
    if (datePicker) {
        datePicker.addEventListener('input', (e) => updateDisplay(e.target.value));
    }

    function changeDay(offset) {
        const currentDate = new Date(datePicker.value + 'T12:00:00Z');
        currentDate.setDate(currentDate.getDate() + offset);
        const newDateStr = currentDate.toISOString().split('T')[0];

        if ((offset > 0 && datePicker.max && newDateStr > datePicker.max) || (offset < 0 && datePicker.min && newDateStr < datePicker.min)) {
            return;
        }
        datePicker.value = newDateStr;
        datePicker.dispatchEvent(new Event('input', { bubbles: true }));
    }

    if (nextDayBtn) {
        nextDayBtn.addEventListener('click', () => changeDay(1));
    }

    if (prevDayBtn) {
        prevDayBtn.addEventListener('click', () => changeDay(-1));
    }
});
