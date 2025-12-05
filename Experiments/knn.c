#include<stdio.h>
#include<math.h>
int main(){

int x[]={1,2,3,6,7,8};
int y[]={2,3,1,5,7,6};
char l[]={'A', 'A', 'A', 'B', 'B', 'B'};
int k=3;
int x1, y1;
struct Pair {
    int d;      // distance
    int index;  // original position
};
int i,dist;
int n = sizeof(x)/sizeof(x[0]);
//printf("%d \n",n);
struct Pair dm[n];
printf("Enter the x1 value: ");
scanf("%d", &x1);
printf("Enter the y1 value: ");
scanf("%d", &y1);

for(i=0; i<=n;i++){
dist = sqrt(pow((x[i]-x1),2)+pow((y[i]-y1),2));
dm[i].d=dist;
dm[i].index=i;
}
/*
printf("Unsorted array: ");
for(i=0; i<=n; i++){ printf("%d ",dm[i].d);printf("%d ",dm[i].index); }
*/
 // bubble sort the struct array by distance
    for(int i = 0; i < n - 1; i++) {
        for(int j = 0; j < n - i - 1; j++) {
            if (dm[j].d > dm[j + 1].d) {

                struct Pair temp = dm[j];
                dm[j] = dm[j + 1];
                dm[j + 1] = temp;
            }
        }
    }
/*
printf("Sorted array: ");
for(i=0; i<=n; i++){printf("%d ",dm[i].d);printf("%d ",dm[i].index);}
*/
// print the first k smallest neighbors
    printf("\nFirst %d neighbors:\n", k);
    for(int i=0; i<k; i++){
        printf("%d) d=%d, index=%d, label=%c\n",
            i+1,
            dm[i].d,
            dm[i].index,
            l[dm[i].index]);
    }
    
    int countA=0, countB=0;
    for(i=0; i<=n; i++){
    char label = l[dm[i].index];
    if(label== 'A') countA++;
    if(label == 'b') countB++;
    }
    
    // print counts
printf("\nCount A = %d\n", countA);
printf("Count B = %d\n", countB);

// determine majority
char predicted;
if(countA > countB) predicted = 'A';
else if(countB > countA) predicted = 'B';
else predicted = '?';  // tie

printf("\nPredicted label = %c\n", predicted);

return 0;
}
