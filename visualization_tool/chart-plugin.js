// Simple heatmap implementation for Chart.js
// This is a workaround since Chart.js doesn't natively support 3D surface plots
// We'll use a heatmap to represent the 3D surface

Chart.register({
    id: 'surface',
    beforeInit: function(chart) {
        const type = chart.config.type;
        
        // Only apply to surface charts
        if (type !== 'surface') return;
        
        // Change the type to scatter
        chart.config.type = 'scatter';
    },
    afterInit: function(chart) {
        const type = chart.config.originalType || chart.config.type;
        
        // Only apply to surface charts
        if (type !== 'surface') return;
        
        // Store original data
        const datasets = chart.data.datasets;
        
        // For each dataset
        datasets.forEach(dataset => {
            if (!dataset.data || !Array.isArray(dataset.data)) return;
            
            // Convert 2D grid data to scatter points with color
            const points = [];
            const colors = [];
            const gridData = dataset.data;
            
            // Find min and max for color scaling
            let min = Infinity;
            let max = -Infinity;
            
            for (let i = 0; i < gridData.length; i++) {
                for (let j = 0; j < gridData[i].length; j++) {
                    const value = gridData[i][j];
                    min = Math.min(min, value);
                    max = Math.max(max, value);
                }
            }
            
            // Create scatter points
            for (let i = 0; i < gridData.length; i++) {
                for (let j = 0; j < gridData[i].length; j++) {
                    const x = j / (gridData[i].length - 1);
                    const y = i / (gridData.length - 1);
                    const value = gridData[i][j];
                    
                    // Normalize value for color
                    const normalizedValue = (value - min) / (max - min || 1);
                    
                    // Add point
                    points.push({
                        x: x,
                        y: y,
                        value: value
                    });
                    
                    // Generate color (using the dataset's color scheme)
                    const hue = dataset.hue || 200;
                    const saturation = dataset.saturation || 70;
                    const lightness = 50 - normalizedValue * 30;
                    colors.push(`hsl(${hue}, ${saturation}%, ${lightness}%)`);
                }
            }
            
            // Update dataset
            dataset.data = points;
            dataset.backgroundColor = colors;
            dataset.borderColor = 'rgba(0,0,0,0.1)';
            dataset.pointRadius = 5;
            dataset.pointHoverRadius = 7;
            
            // Add tooltip callback
            chart.options.plugins = chart.options.plugins || {};
            chart.options.plugins.tooltip = chart.options.plugins.tooltip || {};
            chart.options.plugins.tooltip.callbacks = chart.options.plugins.tooltip.callbacks || {};
            chart.options.plugins.tooltip.callbacks.label = function(context) {
                const point = context.raw;
                return `Value: ${point.value.toFixed(4)}`;
            };
        });
    }
}); 