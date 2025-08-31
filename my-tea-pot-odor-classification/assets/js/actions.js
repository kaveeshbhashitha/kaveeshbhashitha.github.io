document.getElementById('tea-classification-form').addEventListener('submit', function(e) {
      e.preventDefault();
      
      // Show loading state
      const placeholder = document.getElementById('results-placeholder');
      const results = document.getElementById('results-content');
      
      placeholder.innerHTML = `
        <div class="spinner-border text-primary" role="status">
          <span class="visually-hidden">Loading...</span>
        </div>
        <h4 class="mt-3">Analyzing Sample...</h4>
      `;
      
      // Initialize chemical signature chart
      setTimeout(() => {
        const ctx = document.getElementById('chemicalChart').getContext('2d');
        new Chart(ctx, {
          type: 'radar',
          data: {
            labels: ['Theaflavins', 'Thearubigins', 'Caffeine', 'Linalool', 'Geraniol', 'Polyphenols'],
            datasets: [{
              label: 'Your Sample',
              data: [85, 72, 68, 55, 60, 78],
              backgroundColor: 'rgba(54, 162, 235, 0.2)',
              borderColor: 'rgba(54, 162, 235, 1)',
              pointBackgroundColor: 'rgba(54, 162, 235, 1)'
            }]
          },
          options: {
            scales: {
              r: {
                angleLines: { display: true },
                suggestedMin: 0,
                suggestedMax: 100
              }
            }
          }
        });
      }, 100);

      // Simulate API call to your ML model
      setTimeout(function() {
        placeholder.style.display = 'none';
        results.style.display = 'block';
        
        // Initialize tooltips
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function (tooltipTriggerEl) {
          return new bootstrap.Tooltip(tooltipTriggerEl);
        });
        
        // Scroll to results
        document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
      }, 2000);
    });
    
    // Reset form functionality
    document.querySelector('.btn-primary').addEventListener('click', function() {
      document.getElementById('tea-classification-form').reset();
      document.getElementById('results-placeholder').style.display = 'block';
      document.getElementById('results-content').style.display = 'none';
      document.getElementById('results-placeholder').innerHTML = `
        <i class="bi bi-cup-hot results-icon display-1 text-muted"></i>
        <h4 class="mt-3">Submit sample data to view analysis results</h4>
        <p class="text-muted">Your tea classification will appear here</p>
      `;
    });

    // File upload functionality
    document.querySelector('.upload-area').addEventListener('click', function() {
      document.getElementById('batchFile').click();
    });

    document.getElementById('batchFile').addEventListener('change', function(e) {
      if (this.files.length > 0) {
        const fileName = this.files[0].name;
        const uploadArea = document.querySelector('.upload-area');
        uploadArea.innerHTML = `
          <i class="bi bi-file-earmark-spreadsheet display-4 text-success mb-3"></i>
          <p class="mb-3">${fileName}</p>
          <button class="btn btn-success">Analyze Batch</button>
        `;
      }
    });

    //explort results
    document.getElementById("exportReportBtn").addEventListener("click", function () {
        const resultsSection = document.getElementById("results");

        // Show loading indicator if needed
        this.innerText = "Generating PDF..."; 
        this.disabled = true;

        domtoimage.toPng(resultsSection)
            .then(function (dataUrl) {
                const { jsPDF } = window.jspdf;
                const pdf = new jsPDF('p', 'pt', 'a4');
                const imgProps = pdf.getImageProperties(dataUrl);
                const pdfWidth = pdf.internal.pageSize.getWidth();
                const pdfHeight = (imgProps.height * pdfWidth) / imgProps.width;

                pdf.addImage(dataUrl, 'PNG', 0, 0, pdfWidth, pdfHeight);
                pdf.save('Tea_Sample_Report.pdf');

                // Reset button
                document.getElementById("exportReportBtn").innerText = "Export Report";
                document.getElementById("exportReportBtn").disabled = false;
            })
            .catch(function (error) {
                console.error('PDF generation failed:', error);
                alert("Failed to generate PDF");
                document.getElementById("exportReportBtn").innerText = "Export Report";
                document.getElementById("exportReportBtn").disabled = false;
            });
    });
    document.getElementById('selectFileBtn').addEventListener('click', () => {
    document.getElementById('batchFile').click();
});

  document.getElementById('batchFile').addEventListener('change', function(e) {
      const file = e.target.files[0];
      if (!file) return;

      const reader = new FileReader();
      reader.onload = function(e) {
          const text = e.target.result;
          const rows = text.split(/\r?\n/).filter(row => row.trim() !== "");
          if(rows.length === 0) return;

          // Clear previous table
          const tableHead = document.getElementById('csvTableHead');
          const tableBody = document.getElementById('csvTableBody');
          tableHead.innerHTML = "";
          tableBody.innerHTML = "";

          // Add table header
          const headers = rows[0].split(",");
          const headerRow = document.createElement("tr");
          headers.forEach(header => {
              const th = document.createElement("th");
              th.textContent = header.trim();
              headerRow.appendChild(th);
          });
          tableHead.appendChild(headerRow);

          // Add table rows
          for(let i=1; i<rows.length; i++){
              const row = rows[i].split(",");
              const tr = document.createElement("tr");
              row.forEach(cell => {
                  const td = document.createElement("td");
                  td.textContent = cell.trim();
                  tr.appendChild(td);
              });
              tableBody.appendChild(tr);
          }

          // Show the preview table
          document.getElementById('csvPreview').style.display = "block";
      };

      reader.readAsText(file);
  });